import os
import shutil
import argparse
import yaml

import condorTemplates
import pythonTemplates
import helpers
import subprocess

def parseDataset(name, dataset):
    parsed = {}
    
    parsed["name"] = name
    parsed["trackSelection"] = dataset["trackSelection"]
    
    if "fileList" in dataset: # in this case, a fileList is provided, so it's not necessary to create one (or multiple in case of multiple IOVs)
        parsed["dataFrom"] = "fileList"
        parsed["fileList"] = dataset["fileList"]
    else: # in this case, fileLists have to be created using dasgoclient
        parsed["dataFrom"] = "das"
        parsed["dataset"] = dataset["dataset"]
        parsed["json"] = None
        if "json" in dataset:
            parsed["json"] = dataset["json"]
        parsed["lastRun"] = None
        if "lastRun" in dataset:
            parsed["lastRun"] = dataset["lastRun"]
    
    
    parsed["redo"] = False
    if "redo" in dataset:
        parsed["redo"] = dataset["redo"]

    parsed["globalTag"] = dataset["globalTag"]
    if "conditions" in dataset:
        parsed["conditions"] = helpers.parseConditions(dataset["conditions"])
    else:
        parsed["conditions"] = []
    
    parsed["isCosmics"] = False
    if "isCosmics" in dataset:
        parsed["isCosmics"] = dataset["isCosmics"]
        
    parsed["maxEvents"] = -1
    if "maxEvents" in dataset:
        parsed["maxEvents"] = dataset["maxEvents"]
       
    parsed["maxFileSize"] = 350000 # 350MB
    if "maxFileSize" in dataset:
        parsed["maxFileSize"] = dataset["maxFileSize"]
    
    parsed["targetPath"] = dataset["targetPath"]
    parsed["iovName"] = {}
    
    parsed["iovs"] = dataset["iovs"]
    parsed["finished"] = {}
    for iov in dataset["iovs"]:
        parsed["finished"][iov] = False
        
        parsed["iovName"][iov] = "{name}_iov{iov}".format(name=name, iov=iov)
        
        # check if there are already files in the target path with the target name
        # wont use the file list for later, as the number of files has to determined 
        # in a later job anyway for cases where the skim wasnt already performed
        finished = helpers.findFiles(parsed["targetPath"], "{iovname}_{number}.root".format(iovname=parsed["iovName"][iov], number="{number}") )
        if len(finished) != 0: 
            if dataset["redo"]: # the existing files for this iov will be removed later
                pass
            else:  # this iov does not have to be skimmed again, we are done
                print("Found existing skim output files for dataset {} and redo=False, so the skim will not be performed".format(parsed["iovName"][iov]))
                parsed["finished"][iov] = True
                
    return parsed

def parseBaseline(name, baseline):
    parsed = {}
    parsed["name"] = name
    
    parsed["complete"] = False
    if "complete" in baseline:
        parsed["complete"] = baseline["complete"]
        if parsed["complete"]: # no further arguments needed as no reprocessing is performed
            return parsed 
    
    parsed["globalTag"] = baseline["globalTag"]

    if "conditions" in baseline:
        parsed["conditions"] = helpers.parseConditions(baseline["conditions"])
    else:
        parsed["conditions"] = []
       
    parsed["maxEvents"] = -1
    if "maxEvents" in baseline:
        parsed["maxEvents"] = baseline["maxEvents"]
        
    parsed["dataset"] = baseline["dataset"]
    
    return parsed

def parseMeasurement(name, measurement):
    parsed = {}
    parsed["name"] = name
    
    parsed["globalTag"] = measurement["globalTag"]

    if "conditions" in measurement:
        parsed["conditions"] = helpers.parseConditions(measurement["conditions"])
    else:
        parsed["conditions"] = []
    
    parsed["maxIterations"] = 15
    if "maxIterations" in measurement:
        parsed["maxIterations"] = measurement["maxIterations"]
    
    parsed["maxEvents"] = -1
    if "maxEvents" in measurement:
        parsed["maxEvents"] = measurement["maxEvents"]
    
    parsed["baseline"] = measurement["baseline"]
    parsed["dataset"] = measurement["dataset"]
    return parsed


def createConditions(base, dataset, measurement = None):
    # combine conditions defined in dataset (and measurement) and remove double counting
    allConditions = []
    allConditions += dataset["conditions"]
    if measurement is not None:
        allConditions += measurement["conditions"]
    allConditions = list({v['record']:v for v in allConditions}.values())
    
    
    for iov in dataset["iovs"]:
        if measurement is not None:
            if "baseline" in measurement:
                baseName = "measurement_{}_iov{}".format(measurement["name"], iov) # in this case it's a measurement and we might have several dataset IOVs
            else:
                baseName = "measurement_{}".format(measurement["name"]) # in this case it's a baseline and we have only one IOV; the IOV will not be in the name
        else:
            baseName = "dataset_{}".format(dataset["iovName"][iov]) # in this case we have only a dataset
            
        fileName =baseName + "_cff.py"
        with open(os.path.join(base,"src/Alignment/APEEstimation/python/conditions", fileName), "w") as condFile:
            condFile.write(pythonTemplates.conditionsFileHeader)
            
            for condition in allConditions:
                condFile.write( pythonTemplates.conditionsTemplate.format(record=condition["record"], source=condition["source"], tag=condition["tag"]) )
 
 
def createFileList(dataset, workingArea):
    json = ""
    if dataset["json"] is not None:
        json = "--json {}".format(dataset["json"])
    
    iovs = ""
    for iov in dataset["iovs"]:
        iovs += "--iov {} ".format(iov)
        
    if dataset["lastRun"] is not None:
        # every file for successive runs will be put into this iov, which will not be used
        iovs += "--iov {}".format(int(dataset["lastRun"])+1) 
    
    datasetName = dataset["dataset"].replace("/", "_")[1:]
    
    # check if dataset is MC or data:
    import Utilities.General.cmssw_das_client as cmssw_das_client
    # this checks if the only run in this data set is 1, which is only true for MC
    if subprocess.check_output("dasgoclient --query='run dataset={}' --limit=99999".format(dataset["dataset"], limit = 0), shell=True).decode().strip() == "1":
        
        # for MC, we cannot use the script that is used for data, so we have to create the filelist ourselves
        # but this is easy because no json need be applied and only one IOV is used as only one run exists
        files = subprocess.check_output("dasgoclient --query='file dataset={}' --limit=99999".format(dataset["dataset"], limit = 0), shell=True).decode().strip()
        
        rawList = ""
        for fi in files.split("\n"):
            rawList += "'{}',\n".format(fi)
        
        helpers.ensurePathExists(os.path.join(workingArea,datasetName))
        with open(os.path.join(workingArea,datasetName, "Dataset_Alignment_{}_since1_cff.py".format(datasetName,"{}")), "w") as fileList:
            from pythonTemplates import fileListTemplate
            fileList.write(fileListTemplate.format(files=rawList))
        
    else:
        # this script is in Alignment/CommonAlignment/scripts
        # For data, the file lists split into IOVs can be produced with this script
        os.system("tkal_create_file_lists.py {json} -i {dataset} {iovs} -n 9999999 -f 1 -o {workingArea} --force".format(json=json, iovs=iovs, dataset=dataset["dataset"], workingArea=workingArea))
    
    
    dataset["fileList"] = os.path.join(workingArea,datasetName, "Dataset_Alignment_{}_since{}_cff.py".format(datasetName,"{}"))


def main():
    parser = argparse.ArgumentParser(description="Automatically run APE measurements")
    parser.add_argument("-c", "--config", action="store", dest="config", default="config.yaml",
                          help="Config file that configures measurement")
    parser.add_argument("--dryRun", action="store_true", dest="dryRun", default=False,
                          help="Only creates the DAGman files but does not start jobs.")
    args = parser.parse_args()
    
    with open(args.config, "r") as configFile:
        try:
            config_loaded = yaml.safe_load(configFile)
        except yaml.YAMLError as exc:
            print(exc)
    
    if not "workingArea" in config_loaded:
            workingArea = os.getcwd()
    else:
            workingArea = config_loaded["workingArea"]
    
    base = os.environ['CMSSW_BASE']
    
    
    
    # parse config
    parsed_datasets = {}
    parsed_baselines = {}
    parsed_measurements = {}
        
    datasets = config_loaded["datasets"]
    for dataset in datasets:
        parsed = parseDataset(dataset, datasets[dataset]) 
        parsed_datasets[dataset] = parsed
        #checks if all IOVs are finished. If True for every IOV, no skim will be needed and no fileList need be generated
        all_finished = [parsed["finished"][iov] for iov in parsed["iovs"]]
        if parsed["dataFrom"] == "das" and (False in all_finished):
            createFileList(parsed, workingArea)
    
    if "baselines" in config_loaded:
        baselines = config_loaded["baselines"]
        for baseline in baselines:
            # ~ print(baseline)
            parsed = parseBaseline(baseline, baselines[baseline])
            parsed_baselines[baseline] = parsed
    else:
        baselines = {} #  it is legitimate to not have baselines if only datasets are defined
        
    if "measurements" in config_loaded:
        measurements = config_loaded["measurements"]
        for measurement in measurements:
            # ~ print(measurement)
            parsed = parseMeasurement(measurement, measurements[measurement]) 
            parsed_measurements[measurement] = parsed
    else:
        measurements = {} # it is legitimate to not have measurements if one only wants to do baselines or datasets
    
    # check for validity
    # (-> plots need baselines or measurements)
    # -> measurements need baselines
    # -> measurements and baselines need datasets
    #   -> baselines need MC datasets with exactly 1 IOV
    
    for name, measurement in parsed_measurements.items():
        if not measurement["baseline"] in parsed_baselines:
            print("Measurement {} has baseline {} defined, which is not in the configuration.".format(measurement["name"], measurement["baseline"]))
        if not measurement["dataset"] in parsed_datasets:
            print("Measurement {} has dataset {} defined, which is not in the configuration.".format(measurement["name"], measurement["dataset"]))
    
    for name, baseline in parsed_baselines.items():
        if baseline["complete"]:
            continue # no checks to be performed, this measurement is already completed and will not be rerun. it only exists to be referenced by a measurement
        if not baseline["dataset"] in parsed_datasets:
            print("Baseline {} has dataset {} defined, which is not in the configuration.".format(baseline["name"], baseline["dataset"]))
            continue
        if not (len(parsed_datasets[baseline["dataset"]]["iovs"]) == 1):
            print("Dataset {} for baseline {} needs exactly one IOV".format(baseline["dataset"], name))
            
    
    # create files that run jobs
    # -> Skimming (if needed) including renaming and transfer for each IOV of each dataset
    
    
    master_dag_name = os.path.join(workingArea, "main_dag.dag")
    with open(master_dag_name, "w") as master_dag: 
        master_dag.write("# main submission script\n")
        master_dag.write("# dataset jobs\n")
    
    
    for name, dataset in parsed_datasets.items():
        createConditions(base, dataset)
        for iov in dataset["iovs"]:
            if not dataset["finished"][iov]:
                skimSubName = os.path.join(workingArea,"skim_{}.sub".format(dataset["iovName"][iov]))
                with open(skimSubName, "w") as skimSubScript:                  
                    skim_args = "fileList={fileList} outputName={outputName} trackSelection={trackSelection} globalTag={globalTag} maxEvents={maxEvents} maxFileSize={maxFileSize}".format(
                                                                                fileList=dataset["fileList"].format(iov), 
                                                                                outputName=dataset["iovName"][iov],
                                                                                trackSelection=dataset["trackSelection"],
                                                                                globalTag=dataset["globalTag"],
                                                                                maxEvents=dataset["maxEvents"],
                                                                                maxFileSize=dataset["maxFileSize"])
                    skimSubScript.write(condorTemplates.skimSubTemplate.format(workingArea=workingArea, base=base, args=skim_args, target=dataset["targetPath"], name=dataset["iovName"][iov]))
                with open(master_dag_name, "a") as master_dag:
                    master_dag.write("JOB {} {}\n".format("skim_{}".format(dataset["iovName"][iov]), skimSubName))
    
    with open(master_dag_name, "a") as master_dag:
        master_dag.write("\n# baseline subdags and conditions\n")
    
    # -> Baselines
    # -> Handled by prep job
    for name, baseline in parsed_baselines.items():
        if baseline["complete"]:
            continue
        
        dataset = parsed_datasets[baseline["dataset"]]
        iov = dataset["iovs"][0]
        createConditions(base, dataset,baseline)
        
        helpers.ensurePathExists(os.path.join(workingArea, name))
        
        # baseline preparation job
        prep_job_name = os.path.join(workingArea, name, "prep.sub")
        sub_dag_name = os.path.join(workingArea, name, "baseline.dag")
        sub_dag_job = "baseline_{}".format(name)
        with open(prep_job_name, "w") as prep_job:
            prep_job.write(
                condorTemplates.prepSubTemplate.format(base=base,
                                                    workingArea=workingArea,
                                                    globalTag=baseline["globalTag"],
                                                    measName=name,
                                                    isCosmics=dataset["isCosmics"],
                                                    maxIterations=0,
                                                    baselineName=name,
                                                    dataDir=dataset["targetPath"],
                                                    fileName=dataset["iovName"][iov],
                                                    maxEvents=baseline["maxEvents"],
                                                    isBaseline=True)
            )
                
  
  
  
  
  
        
        with open(master_dag_name, "a") as master_dag:
            master_dag.write("JOB prep_{} {}\n".format(name, prep_job_name))
            
            iov = dataset["iovs"][0] # only 1 IOV for baseline measurements
            if not dataset["finished"][iov]: # if dataset is already finished, there will be no job to wait for
                master_dag.write("PARENT {} CHILD prep_{}\n".format("skim_{}".format(dataset["iovName"][iov]),name))
            
            master_dag.write("SUBDAG EXTERNAL {} {}\n".format(sub_dag_job, sub_dag_name))
            master_dag.write("PARENT prep_{} CHILD {}\n".format(name, sub_dag_job))
                    
                    
        # create subdag file, only 1 for baseline because only 1 IOV
        with open(sub_dag_name, "w") as sub_dag:
            sub_dag.write("# Will be filled later\n")
    
    with open(master_dag_name, "a") as master_dag:
        master_dag.write("\n# measurement subdags and conditions\n")
        
    # -> Measurements
    # -> Handled by prep job
    for name, measurement in parsed_measurements.items():
        dataset = parsed_datasets[measurement["dataset"]]
        baseline = parsed_baselines[measurement["baseline"]]
        baseline_dag_name = "baseline_{}".format(baseline["name"]) 
        
        createConditions(base, parsed_datasets[measurement["dataset"]],measurement)
        
        for iov in dataset["iovs"]:
            meas_name = "{}_iov{}".format(name, iov)
            helpers.ensurePathExists(os.path.join(workingArea, meas_name))
            helpers.newIterFolder(workingArea, meas_name, "apeObjects")
            
            prep_job_name = os.path.join(workingArea, meas_name, "prep.sub")
            sub_dag_name = os.path.join(workingArea, meas_name, "measurement.dag")
            sub_dag_job = "measurement_{}".format(meas_name)
            
            with open(prep_job_name, "w") as prep_job:
                prep_job.write(
                condorTemplates.prepSubTemplate.format(base=base,
                                                    workingArea=workingArea,
                                                    globalTag=measurement["globalTag"],
                                                    measName=meas_name,
                                                    isCosmics=dataset["isCosmics"],
                                                    maxIterations=measurement["maxIterations"],
                                                    baselineName=baseline["name"],
                                                    dataDir=dataset["targetPath"],
                                                    fileName=dataset["iovName"][iov],
                                                    maxEvents=measurement["maxEvents"],
                                                    isBaseline=False)
                )
            
            with open(master_dag_name, "a") as master_dag:
                master_dag.write("JOB prep_{} {}\n".format(meas_name, prep_job_name))
                
                if not dataset["finished"][iov]: # if dataset is already finished, there will be no job to wait for
                    master_dag.write("PARENT {} CHILD prep_{}\n".format("skim_{}".format(dataset["iovName"][iov]),meas_name))
                
                master_dag.write("SUBDAG EXTERNAL {} {}\n".format(sub_dag_job, sub_dag_name))
                master_dag.write("PARENT prep_{} CHILD {}\n".format(meas_name, sub_dag_job))
                if not baseline["complete"]: # if this has to be run, then we have to wait for it to finish first before starting the measurement
                    master_dag.write("PARENT {} CHILD {}\n".format(baseline_dag_name, sub_dag_job))
                    
            with open(sub_dag_name, "w") as sub_dag:
                sub_dag.write("# Will be filled later\n")
    
    if not args.dryRun:
        subprocess.call("condor_submit_dag {}".format(master_dag_name), shell=True)
if __name__ == "__main__":
    main()
-- dummy change --
