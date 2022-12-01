import os
import copy

def GCP(config, validationDir):
    ## List with all jobs
    jobs = []

    ## Get unit test flag
    doUnitTest = False 
    if "doUnitTest" in config["validations"]["GCP"].keys():
        doUnitTest = config["validations"]["GCP"]["doUnitTest"]

    ## Main loop
    for comparison in config["validations"]["GCP"]["compare"]:
        for ali_pair in config["validations"]["GCP"]["compare"][comparison]:
            ref_name  = copy.deepcopy(config["validations"]["GCP"]["compare"][comparison][ali_pair]["reference"])
            comp_name = copy.deepcopy(config["validations"]["GCP"]["compare"][comparison][ali_pair]["compared"])
            IOVpair_list = []
            IOVali_list = []

            # Construct pairs from IOVlist
            IOV_list = []
            if "IOVlist" in config["validations"]["GCP"]["compare"][comparison][ali_pair]: IOV_list = copy.deepcopy(config["validations"]["GCP"]["compare"][comparison][ali_pair]["IOVlist"])
            IOV_list.sort()
            for idx,IOV in enumerate(IOV_list):
                if ref_name == comp_name:
                    IOV_pair = str(IOV)+'_vs_'+str(IOV_list[0])
                    IOV_ali_r = ref_name+'_'+str(IOV_list[0])
                    IOV_ali_c = comp_name+'_'+str(IOV)
                else:
                    IOV_pair = str(IOV)+'_vs_'+str(IOV)
                    IOV_ali_r = ref_name+'_'+str(IOV)
                    IOV_ali_c = comp_name+'_'+str(IOV)
                if IOV_pair not in IOVpair_list: IOVpair_list.append(IOV_pair)
                if IOV_ali_r not in IOVali_list: IOVali_list.append(IOV_ali_r)
                if IOV_ali_c not in IOVali_list: IOVali_list.append(IOV_ali_c)

            # Read explicit pairs from IOVpairs
            pair_list = []
            if "IOVpairs" in config["validations"]["GCP"]["compare"][comparison][ali_pair]: pair_list = copy.deepcopy(config["validations"]["GCP"]["compare"][comparison][ali_pair]["IOVpairs"])
            for IOV_p in pair_list:
                IOV_pair = str(IOV_p[0])+'_vs_'+str(IOV_p[1])
                IOV_ali_r = ref_name+'_'+str(IOV_p[1])
                IOV_ali_c = comp_name+'_'+str(IOV_p[0])
                if IOV_pair not in IOVpair_list: IOVpair_list.append(IOV_pair)
                if IOV_ali_r not in IOVali_list: IOVali_list.append(IOV_ali_r)
                if IOV_ali_c not in IOVali_list: IOVali_list.append(IOV_ali_c)

            # GCP Ntuple job preps
            for IOV_ali in IOVali_list:
                ali = IOV_ali.split('_')[0]
                IOV = int(IOV_ali.split('_')[1])
                workDir = "{}/GCP/{}/{}/{}".format(validationDir, comparison, 'Ntuples', IOV_ali)
                
                # local config 
                local = {}
                local["output"] = "{}/{}/{}/{}/{}".format(config["LFS"], config["name"], comparison, 'Ntuples', IOV_ali)
                local["alignments"] = copy.deepcopy(config["alignments"][ali])
                local["validation"] = {}
                local["validation"]['GCP'] = copy.deepcopy(config["validations"]["GCP"][comparison])
                local["validation"]['GCP']['doUnitTest'] = doUnitTest 
                local["validation"]['IOV'] = IOV

                # job info
                job = {
                    "name": "GCP_{}_Ntuple_{}".format(comparison, IOV_ali),
                    "dir": workDir,
                    "exe": "cmsRun",
                    "cms-config": "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/GCP_Ntuples_cfg.py".format(os.environ["CMSSW_BASE"]),
                    "run-mode": "Condor",
                    "dependencies": [],
                    "config": local,
                    "flavour": "espresso", 
                }

                # Ntuple jobs might appear multiple times, only add if not there yet
                already_there = False
                for j in jobs:
                    if j["name"] == job["name"]: 
                        already_there = True
                        break

                if not already_there: jobs.append(job)

            # Comparison job preps
            for IOV_pair in IOVpair_list:
                ref_IOV  = int(IOV_pair.split('_vs_')[1])
                comp_IOV = int(IOV_pair.split('_vs_')[0])
               
                # local config
                local = {} 
                local["output"] = "{}/{}/{}/{}/{}".format(config["LFS"], config["name"], comparison, ali_pair, IOV_pair)
                local["alignments"] = {}
                local["alignments"]["ref"]  = copy.deepcopy(config["alignments"][ref_name])
                local["alignments"]["comp"] = copy.deepcopy(config["alignments"][comp_name])
                local["validation"] = {}
                local["validation"]['GCP'] = copy.deepcopy(config["validations"]["GCP"][comparison])
                local["validation"]['GCP']['doUnitTest'] = doUnitTest
                local["validation"]["IOVref"] = ref_IOV
                local["validation"]["ALIref"] = ref_name
                local["validation"]["IOVcomp"] = comp_IOV
                local["validation"]["ALIcomp"] = comp_name

                # dependancies
                parents = []
                for j in jobs:
                    if not comparison in j['name']: continue
                    if not 'Ntuple' in j['name']: continue
                    if ref_name in j['name'] and str(ref_IOV) in j['name']: 
                        parents.append(j['name'])
                        local["input_ref"] = j['config']['output']
                    if comp_name in j['name'] and str(comp_IOV) in j['name']: 
                        parents.append(j['name'])
                        local["input_comp"] = j['config']['output']
                
                # Comparison jobs
                for step in ['GCPtree', 'GCPcpp', 'GCPpython']:
                    workDir = "{}/GCP/{}/{}/{}/{}".format(validationDir, comparison, ali_pair, IOV_pair, step)
                    job = {
                        "name": "GCP_{}_{}_{}_{}".format(comparison, ali_pair, IOV_pair, step),
                        "dir": workDir,
                        "run-mode": "Condor",
                        "config": local, 
                        "flavour": "espresso", 
                    }
                    if step == 'GCPtree':
                        job['exe'] = 'cmsRun'
                        job['cms-config'] = "{}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/GCP_tree_cfg.py".format(os.environ["CMSSW_BASE"]) 
                        job['dependencies'] = parents
                    elif step == 'GCPcpp':
                        job['flavour'] = 'microcentury' 
                        job['exe'] = 'GCP'
                        job['dependencies'] = parents + ["GCP_{}_{}_{}_{}".format(comparison, ali_pair, IOV_pair, 'GCPtree')]
                    else: 
                        job['exe'] = 'GCPpyPlots.py'
                        job['dependencies'] = parents + ["GCP_{}_{}_{}_{}".format(comparison, ali_pair, IOV_pair, 'GCPtree')]

                    jobs.append(job)

    return jobs
