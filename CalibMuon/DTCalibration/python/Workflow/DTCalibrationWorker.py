from CalibMuon.DTCalibration.Workflow.DTTTrigProd import DTTTrigProd
from CalibMuon.DTCalibration.Workflow.DTTTrigTimeBoxesWriter import DTTTrigTimeBoxesWriter
from CalibMuon.DTCalibration.Workflow.DTResidualCalibration import DTResidualCalibration
from CalibMuon.DTCalibration.Workflow.DTTTrigResidualCorr import DTTTrigResidualCorr 
from CalibMuon.DTCalibration.Workflow.DTTTrigValid import DTTTrigValid
from CalibMuon.DTCalibration.Workflow.DTValidSummary import DTValidSummary
from CalibMuon.DTCalibration.Workflow.DTVDriftSegmentCalibration import DTVDriftSegmentCalibration
from CalibMuon.DTCalibration.Workflow.DTVDriftSegmentWriter import DTVDriftSegmentWriter
from CalibMuon.DTCalibration.Workflow.DTVDriftMeanTimerCalibration import DTVDriftMeanTimerCalibration
from CalibMuon.DTCalibration.Workflow.DTVDriftMeanTimerWriter import DTVDriftMeanTimerWriter
from CalibMuon.DTCalibration.Workflow.DTNoiseCalibration import DTNoiseCalibration
from CalibMuon.DTCalibration.Workflow.DTDQMValidation import DTDQMValidation
from CalibMuon.DTCalibration.Workflow.DTDQMMerge import DTDQMMerge
from CalibMuon.DTCalibration.Workflow.DTDQMHarvesting import DTDQMHarvesting
from CalibMuon.DTCalibration.Workflow.DTDqm import DTDqm
from CalibMuon.DTCalibration.Workflow.DTT0DBValidation import DTT0DBValidation
from CalibMuon.DTCalibration.Workflow.DTAnalysisResiduals import DTAnalysisResiduals
from CalibMuon.DTCalibration.Workflow.CrabWatch import CrabWatch
from CalibMuon.DTCalibration.Workflow.tools import listFilesInCastor,haddInCastor,listFilesLocal,haddLocal,copyFilesFromCastor,copyFilesLocal,parseInput,getDatasetStr
import sys,os,time,optparse

class DTCalibrationWorker:
    def __init__(self,run,config):
        self.config = config
        self.refRun = run

    def run(self,type,mode,execute):
        config = None
        if self.config: config = self.config
        refRun = 1
        if self.refRun: refRun = self.refRun

	if   type == 'ttrig':        self.runTtrigWorkflow(mode,refRun,config,execute)
	elif type == 'vdrift':       self.runVDriftWorkflow(mode,refRun,config,execute)
	elif type == 'noise':        self.runNoiseWorkflow(mode,refRun,config,execute)
	elif type == 't0':           self.runT0Workflow(mode,refRun,config,execute)
	elif type == 'validation':   self.runValidationWorkflow(mode,refRun,config,execute)
	elif type == 'analysis':     self.runAnalysisWorkflow(mode,refRun,config,execute)
	elif type == 'dbvalidation':
            inputFiles = []
            if config.dbFiles: inputFiles = config.dbFiles
            self.runDBValidationWorkflow(mode,refRun,inputFiles,config,execute)

        return 0

    def dqmOutputDir(self,type,dirLabel,config):
	dqm_output_dir = ''
	if config.stageOutLocal:
	    from crab_util import findLastWorkDir
	    cwd = os.getcwd()
	    crab_task_dir = config.base_dir + '/' + dirLabel
	    os.chdir(crab_task_dir)
	    crabdir = findLastWorkDir('crab_0_')
	    if not crabdir: raise RuntimeError,'Could not find CRAB dir in %s' % crab_task_dir
	    os.chdir(cwd)
	    dqm_output_dir = crabdir + "/res"
	elif config.stageOutCAF:
	    datasetstr = getDatasetStr(config.datasetpath)
	    dqm_output_dir = config.castorpath + '/DTCalibration/' + datasetstr + '/Run' + str(run) + '/' + type + '/' + dirLabel + '/' + 'v' + str(trial)

	return dqm_output_dir

    def runTtrigProd(self,run,runselection,trial,config,runStep=True):

	print "Processing tTrig production"
	#config.runselection = runselection
	datasetstr = getDatasetStr(config.datasetpath)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/TTrigCalibration/' + 'TimeBoxes' + '/' + 'v' + str(trial)
       
	task_dir = config.base_dir + '/TimeBoxes'
	dtTtrigProd = DTTTrigProd(run,task_dir,config) 
	dtTtrigProd.writeCfg()
	
	if runStep:
	    project_prod = dtTtrigProd.run()

	    print "Sent calibration jobs with project",project_prod
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold
	  
	    crabProd = CrabWatch(project_prod)
	    crabProd.setThreshold(config.jobsFinishedThreshold)
	    crabProd.start()
	    crabProd.join()

	    result_file = config.result_dir + '/DTTimeBoxes_%s.root'%run
	    if config.stageOutLocal:
		output_dir = project_prod + "/res"
		haddLocal(output_dir,result_file) 
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		haddInCastor(castor_dir,result_file,'DTTimeBoxes','rfio://castorcms/','?svcClass=cmscafuser')

	    return project_prod

	return None

    def runTtrigWriter(self,run,config,runStep=True):

	print "Processing tTrig correction"
	dtTtrigWriter = DTTTrigTimeBoxesWriter(run,config.run_dir,config.result_dir,config)
	dtTtrigWriter.writeCfg()
     
	if runStep:  
	    dtTtrigWriter.run()

	    print "Finished processing:"
	    for pset in dtTtrigWriter.configs: print "--->",pset

    def runResidualCalib(self,run,runselection,trial,input_db,label,result_file,config,runStep=True):

	print "Processing tTrig calibration"
	#config.runselection = runselection
	datasetstr = getDatasetStr(config.datasetpath)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/TTrigCalibration/' + label + '/' + 'v' + str(trial)
	
	task_dir = config.base_dir + '/' + label
	dtResidualCalib = DTResidualCalibration(run=run,
						dir=task_dir,
						input_db=input_db,
						config=config) 
	dtResidualCalib.writeCfg()

	if runStep:
	    project_residualCalib = dtResidualCalib.run()

	    print "Sent calibration jobs with project",project_residualCalib
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabResidualCalib = CrabWatch(project_residualCalib)
	    crabResidualCalib.setThreshold(config.jobsFinishedThreshold)
	    crabResidualCalib.start()
	    crabResidualCalib.join()

	    if config.stageOutLocal:
		output_dir = project_residualCalib + "/res"
		haddLocal(output_dir,result_file,'residuals')
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		haddInCastor(castor_dir,result_file,'residuals','rfio://castorcms/','?svcClass=cmscafuser')

	    return project_residualCalib

	return None

    def runAnalysisResiduals(self,run,runselection,trial,label,result_file,config,runStep=True):

	print "Processing residuals analysis"
	datasetstr = getDatasetStr(config.datasetpath)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/AnalysisResiduals/' + label + '/' + 'v' + str(trial)
	
	task_dir = config.base_dir + '/' + label
	dtAnalysisResiduals = DTAnalysisResiduals(run=run,
				                  dir=task_dir,
					          config=config) 
	dtAnalysisResiduals.writeCfg()

	if runStep:
	    project_analysisResiduals = dtAnalysisResiduals.run()

	    print "Sent jobs with project",project_analysisResiduals
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabAnalysisResiduals = CrabWatch(project_analysisResiduals)
	    crabAnalysisResiduals.setThreshold(config.jobsFinishedThreshold)
	    crabAnalysisResiduals.start()
	    crabAnalysisResiduals.join()

	    if config.stageOutLocal:
		output_dir = project_analysisResiduals + "/res"
		haddLocal(output_dir,result_file,'residuals')
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		haddInCastor(castor_dir,result_file,'residuals','rfio://castorcms/','?svcClass=cmscafuser')

	    return project_analysisResiduals

	return None

    def runTtrigResidualCorr(self,run,input_db,root_file,config,runStep=True):

	print "Processing tTrig residual correction"
	dtTtrigResidualCorr = DTTTrigResidualCorr(run=run,
						  dir=config.run_dir,
						  input_db=input_db,
						  residuals=root_file,
						  result_dir=config.result_dir,
						  config=config)
	dtTtrigResidualCorr.writeCfg()

	if runStep:  
	    dtTtrigResidualCorr.run()

	    print "Finished processing:"
	    for pset in dtTtrigResidualCorr.configs: print "--->",pset

    def runTtrigValid(self,run,runselection,trial,input_db,label,config,runStep=True):

	print "Processing tTrig validation"
	#config.runselection = runselection
	datasetstr = getDatasetStr(config.datasetpath)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/TTrigValidation/' + label + '/' + 'v' + str(trial)

	task_dir = config.base_dir + '/' + label
	dtTtrigValid = DTTTrigValid(run=run,
				    dir=task_dir,
				    input_db=input_db,
				    config=config)
	dtTtrigValid.writeCfg()

	if runStep:
	    project_valid = dtTtrigValid.run()

	    print "Sent validation jobs with project",project_valid
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabValid = CrabWatch(project_valid)
	    crabValid.setThreshold(config.jobsFinishedThreshold)
	    crabValid.start()
	    crabValid.join()

	    """  
	    if config.stageOutLocal:
		output_dir = project_valid + "/res"
		haddLocal(output_dir,result_file,'residuals')
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		haddInCastor(castor_dir,result_file,'residuals','rfio://castorcms/','?svcClass=cmscafuser')
	    """

	    return project_valid

	return None

    def runTtrigValidSummary(self,run,input_file,output_file,config,runStep=True):

	print "Processing Validation Summary"
	dtTtrigValidSummary = DTValidSummary(run,config.run_dir,input_file,output_file,config)
	dtTtrigValidSummary.writeCfg()

	if runStep:
	    dtTtrigValidSummary.run()
     
	    print "...Validation Summary finished"

    def runVDriftSegmentCalib(self,run,runselection,trial,label,result_file,config,runStep=True):

	print "Processing vDrift calibration"
	#config.runselection = runselection
	datasetstr = getDatasetStr(config.datasetpath)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/VDriftCalibration/' + label + '/' + 'v' + str(trial)

	task_dir = config.base_dir + '/' + label
	dtVDriftSegment = DTVDriftSegmentCalibration(run=run,
						     dir=task_dir,
						     config=config)
	dtVDriftSegment.writeCfg()

	if runStep:
	    project_segment = dtVDriftSegment.run()

	    print "Sent validation jobs with project",project_segment
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabVDriftSegment = CrabWatch(project_segment)
	    crabVDriftSegment.setThreshold(config.jobsFinishedThreshold)
	    crabVDriftSegment.start()
	    crabVDriftSegment.join()

	    if config.stageOutLocal:
		output_dir = project_segment + "/res"
		haddLocal(output_dir,result_file,'DTVDriftHistos')
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		haddInCastor(castor_dir,result_file,'DTVDriftHistos','rfio://castorcms/','?svcClass=cmscafuser')

	    return project_segment

	return None

    def runVDriftSegmentWriter(self,run,root_file,config,runStep=True):

	print "Processing vDrift writer"
	dtVDriftSegmentWriter = DTVDriftSegmentWriter(run=run,
						      dir=config.run_dir,
						      input_file=root_file,
						      output_dir=config.result_dir,
						      config=config)
	dtVDriftSegmentWriter.writeCfg()

	if runStep:
	    dtVDriftSegmentWriter.run()

	    print "Finished processing:"
	    for pset in dtVDriftSegmentWriter.configs: print "--->",pset

    def runVDriftMeanTimerCalib(self,run,runselection,trial,label,result_file,config,runStep=True):

	print "Processing vDrift calibration"
	#config.runselection = runselection
	datasetstr = getDatasetStr(config.datasetpath)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/VDriftCalibration/' + label + '/' + 'v' + str(trial)

	task_dir = config.base_dir + '/' + label
	dtVDriftMeanTimer = DTVDriftMeanTimerCalibration(run=run,
							 dir=task_dir,
							 config=config)
	dtVDriftMeanTimer.writeCfg()

	if runStep:
	    project_meantimer = dtVDriftMeanTimer.run()

	    print "Sent validation jobs with project",project_meantimer
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabVDriftMeanTimer = CrabWatch(project_meantimer)
	    crabVDriftMeanTimer.setThreshold(config.jobsFinishedThreshold)
	    crabVDriftMeanTimer.start()
	    crabVDriftMeanTimer.join()

	    if config.stageOutLocal:
		output_dir = project_meantimer + "/res"
		haddLocal(output_dir,result_file,'DTTMaxHistos')
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		haddInCastor(castor_dir,result_file,'DTTMaxHistos','rfio://castorcms/','?svcClass=cmscafuser')

	    return project_meantimer

	return None

    def runVDriftMeanTimerWriter(self,run,root_file,config,runStep=True):

	print "Processing vDrift writer"
	dtVDriftMeanTimerWriter = DTVDriftMeanTimerWriter(run=run,
						      dir=config.run_dir,
						      input_file=root_file,
						      output_dir=config.result_dir,
						      config=config)
	dtVDriftMeanTimerWriter.writeCfg()

	if runStep:
	    dtVDriftMeanTimerWriter.run()

	    print "Finished processing:"
	    for pset in dtVDriftMeanTimerWriter.configs: print "--->",pset

    def runDQMClient(self,run,output_dir,config,runStep=True):

	print "Processing DQM Merge"

	if runStep:
	    dqm_files = [] 
	    if config.stageOutLocal:
		dqm_files = listFilesLocal(output_dir,'DQM')
		dqm_files = ['file:%s' % item for item in dqm_files]
		dtDqmFinal = DTDqm(run,config.run_dir,dqm_files,config.result_dir,config)
		dtDqmFinal.writeCfg()
		dtDqmFinal.run()
	    elif config.stageOutCAF:
		dqm_files = listFilesInCastor(output_dir,'DQM','')
		dqm_files = [file.replace('/castor/cern.ch/cms','') for file in dqm_files] 
		dtDqmFinal = DTDqm(run,config.run_dir,dqm_files,config.result_dir,config)
		dtDqmFinal.writeCfg()
		dtDqmFinal.run()

	    print "...DQM Merge finished"
	else:
	    dqm_files = [] 
	    dtDqmFinal = DTDqm(run,config.run_dir,dqm_files,config.result_dir,config)
	    dtDqmFinal.writeCfg()

    def runDQMHarvesting(self,run,output_dir,config,runStep=True):

	print "Processing DQM harvesting"

	if runStep:
	    dqm_files = [] 
	    if config.stageOutLocal:
		dqm_files = listFilesLocal(output_dir,'DQM')
		dqm_files = ['file:%s' % item for item in dqm_files]
		dtDqmFinal = DTDQMHarvesting(run,config.run_dir,dqm_files,config.result_dir,config)
		dtDqmFinal.writeCfg()
		dtDqmFinal.run()
	    elif config.stageOutCAF:
		dqm_files = listFilesInCastor(output_dir,'DQM','')
		dqm_files = [file.replace('/castor/cern.ch/cms','') for file in dqm_files] 
		dtDqmFinal = DTDQMHarvesting(run,config.run_dir,dqm_files,config.result_dir,config)
		dtDqmFinal.writeCfg()
		dtDqmFinal.run()

	    print "...DQM harvesting finished"
	else:
	    dqm_files = [] 
	    dtDqmFinal = DTDQMHarvesting(run,config.run_dir,dqm_files,config.result_dir,config)
	    dtDqmFinal.writeCfg()

    def runDQMMerge(self,run,output_dir,config,runStep=True):

	print "Processing DQM merge"

	if runStep:
	    dqm_files = [] 
	    if config.stageOutLocal:
		dqm_files = listFilesLocal(output_dir,'DQM')
		dqm_files = ['file:%s' % item for item in dqm_files]
		dtDQMMerge = DTDQMMerge(run,config.run_dir,dqm_files,config.result_dir,config)
		dtDQMMerge.writeCfg()
		dtDQMMerge.run()
	    elif config.stageOutCAF:
		dqm_files = listFilesInCastor(output_dir,'DQM','')
		dqm_files = [file.replace('/castor/cern.ch/cms','') for file in dqm_files] 
		dtDQMMerge = DTDQMMerge(run,config.run_dir,dqm_files,config.result_dir,config)
		dtDQMMerge.writeCfg()
		dtDQMMerge.run()

	    print "...DQM merge finished"
	else:
	    dqm_files = [] 
	    dtDQMMerge = DTDQMMerge(run,config.run_dir,dqm_files,config.result_dir,config)
	    dtDQMMerge.writeCfg()

    ############################################################ 
    # tTrig workflow
    ############################################################ 
    def runTtrigWorkflow(self,mode,run,config,execute=True):
	trial = config.trial
	runselection = config.runselection
	ttrig_input_db = None
	if hasattr(config,'inputTTrigDB') and config.inputTTrigDB: ttrig_input_db = os.path.abspath(config.inputTTrigDB)
	result_dir = config.result_dir
	if mode == 'timeboxes':
	    timeBoxes = os.path.abspath(result_dir + '/' + 'DTTimeBoxes_' + run + '.root')
	    ttrig_timeboxes_db = os.path.abspath(result_dir + '/' + 'ttrig_timeboxes_' + run + '.db')
	    residualsFirst = os.path.abspath(result_dir + '/' + 'DTResidualValidation_' + run + '.root')
	    ttrig_residuals_db = os.path.abspath(result_dir + '/' + 'ttrig_residuals_' + run + '.db')
	    residualsResidCorr = os.path.abspath(result_dir + '/' + 'DTResidualValidation_ResidCorr_' + run + '.root')
	    summaryResiduals = os.path.abspath(result_dir + '/' + 'SummaryResiduals_' + run + '.root') 

	    if not execute:
		print "Writing configuration files.."
		self.runTtrigProd(run,runselection,trial,config,False)
		self.runTtrigWriter(run,config,False)
		self.runResidualCalib(run,runselection,trial,ttrig_timeboxes_db,'Residuals',residualsFirst,config,False)
		self.runTtrigResidualCorr(run,ttrig_timeboxes_db,residualsFirst,config,False)                
		self.runTtrigValid(run,runselection,trial,ttrig_residuals_db,'ResidualsResidCorr',config,False)
		#self.runTtrigValidSummary(run,residualsResidCorr,summaryResiduals,config,False)
		self.runDQMClient(run,'',config,False)

		sys.exit(0)

	    # Produce time-boxes
	    if not os.path.exists(timeBoxes): self.runTtrigProd(run,runselection,trial,config)
	    if not os.path.exists(timeBoxes): raise RuntimeError,'Could not produce %s' % timeBoxes

	    # Write tTrig DB
	    if not os.path.exists(ttrig_timeboxes_db): self.runTtrigWriter(run,config)
	    if not os.path.exists(ttrig_timeboxes_db): raise RuntimeError,'Could not produce %s' % ttrig_timeboxes_db

	    # Produce residuals
	    if not os.path.exists(residualsFirst):
		self.runResidualCalib(run,runselection,trial,ttrig_timeboxes_db,'Residuals',residualsFirst,config)
	    if not os.path.exists(residualsFirst): raise RuntimeError,'Could not produce %s' % residualsFirst

	    # Correction from residuals and write tTrig DB
	    if not os.path.exists(ttrig_residuals_db): self.runTtrigResidualCorr(run,ttrig_timeboxes_db,residualsFirst,config)
	    if not os.path.exists(ttrig_residuals_db): raise RuntimeError,'Could not produce %s' % ttrig_residuals_db

	    # Validation
	    self.runTtrigValid(run,runselection,trial,ttrig_residuals_db,'ResidualsResidCorr',config)

	    """
	    # Summary of validation
	    if not os.path.exists(summaryResiduals): self.runTtrigValidSummary(run,residualsResidCorr,summaryResiduals,config)
	    if not os.path.exists(summaryResiduals): raise RuntimeError,'Could not produce %s' % summaryResiduals
	    """
	    # Produce DQM output 
	    dqm_output_dir = self.dqmOutputDir('TTrigValidation','ResidualsResidCorr',config)
	    self.runDQMClient(run,dqm_output_dir,config)

	elif mode == 'residuals':
	    residualsFirst = os.path.abspath(result_dir + '/' + 'DTResidualValidation_' + run + '.root')
	    ttrig_residuals_db = os.path.abspath(result_dir + '/' + 'ttrig_residuals_' + run + '.db')
	    residualsResidCorr = os.path.abspath(result_dir + '/' + 'DTResidualValidation_ResidCorr_' + run + '.root')
	    summaryResiduals = os.path.abspath(result_dir + '/' + 'SummaryResiduals_' + run + '.root')

	    if not execute:
		print "Writing configuration files.."
		if ttrig_input_db:
		    self.runResidualCalib(run,runselection,trial,ttrig_input_db,'Residuals',residualsFirst,config,False)
		    self.runTtrigResidualCorr(run,ttrig_input_db,residualsFirst,config,False)
		else:
		    self.runResidualCalib(run,runselection,trial,None,'Residuals',residualsFirst,config,False)
		    self.runTtrigResidualCorr(run,None,residualsFirst,config,False)

		self.runTtrigValid(run,runselection,trial,ttrig_residuals_db,'ResidualsResidCorr',config,False)
		#self.runTtrigValidSummary(run,residualsResidCorr,summaryResiduals,config,False)
		self.runDQMClient(run,'',config,False)

		sys.exit(0)

	    # Produce residuals
	    if not os.path.exists(residualsFirst):
		if ttrig_input_db:
		    self.runResidualCalib(run,runselection,trial,ttrig_input_db,'Residuals',residualsFirst,config) 
		else:
		    self.runResidualCalib(run,runselection,trial,None,'Residuals',residualsFirst,config)
	    if not os.path.exists(residualsFirst): raise RuntimeError,'Could not produce %s' % residualsFirst

	    # Correction from residuals and write tTrig DB
	    if not os.path.exists(ttrig_residuals_db):
		if ttrig_input_db: self.runTtrigResidualCorr(run,ttrig_input_db,residualsFirst,config)
		else: self.runTtrigResidualCorr(run,None,residualsFirst,config)
	    if not os.path.exists(ttrig_residuals_db): raise RuntimeError,'Could not produce %s' % ttrig_residuals_db

	    # Validation
	    self.runTtrigValid(run,runselection,trial,ttrig_residuals_db,'ResidualsResidCorr',config)

	    """  
	    # Summary of validation
	    if not os.path.exists(summaryResiduals):
		self.runTtrigValidSummary(run,residualsResidCorr,summaryResiduals,config)
	    if not os.path.exists(summaryResiduals): raise RuntimeError,'Could not produce %s' % summaryResiduals
	    """
	    # Produce DQM output 
	    dqm_output_dir = self.dqmOutputDir('TTrigValidation','ResidualsResidCorr',config)
	    self.runDQMClient(run,dqm_output_dir,config)

	elif mode == 'validation':
	    residualsValid = os.path.abspath(result_dir + '/' + 'DTResidualValidation_' + run + '.root')
	    summaryResiduals = os.path.abspath(result_dir + '/' + 'SummaryResiduals_' + run + '.root')

	    if not execute:
		print "Writing configuration files.."
		if ttrig_input_db:
		    self.runTtrigValid(run,runselection,trial,ttrig_input_db,'Residuals',config,False)
		else:
		    self.runTtrigValid(run,runselection,trial,None,'Residuals',config,False)

		#self.runTtrigValidSummary(run,residualsValid,summaryResiduals,config,False)
		self.runDQMClient(run,'',config,False)

		sys.exit(0)

	    # Validation
	    if ttrig_input_db:
		self.runTtrigValid(run,runselection,trial,ttrig_input_db,'Residuals',config)
	    else:
		self.runTtrigValid(run,runselection,trial,None,'Residuals',config)

	    """
	    # Summary of validation
	    if not os.path.exists(summaryResiduals):
		self.runTtrigValidSummary(run,residualsValid,summaryResiduals,config)
	    if not os.path.exists(summaryResiduals): raise RuntimeError,'Could not produce %s' % summaryResiduals
	    """

	    # Produce DQM output 
	    dqm_output_dir = self.dqmOutputDir('TTrigValidation','Residuals',config)
	    self.runDQMClient(run,dqm_output_dir,config)

	return 0

    ############################################################ 
    # vDrift workflow
    ############################################################
    def runVDriftWorkflow(self,mode,run,config,execute=True):
	trial = config.trial
	runselection = config.runselection
	result_dir = config.result_dir
	if mode == 'segment':
	    vDriftHistos = os.path.abspath(result_dir + '/' + 'DTVDriftHistos_' + run + '.root')
	    vDrift_segment_db = os.path.abspath(result_dir + '/' + 'vDrift_segment_' + run + '.db')

	    if not execute:
		print "Writing configuration files.."
		self.runVDriftSegmentCalib(run,runselection,trial,'VDriftHistos',vDriftHistos,config,False)
		self.runVDriftSegmentWriter(run,vDriftHistos,config,False)
	 
		sys.exit(0)

	    # Produce vDrift histos
	    if not os.path.exists(vDriftHistos):
		self.runVDriftSegmentCalib(run,runselection,trial,'VDriftHistos',vDriftHistos,config)
	    if not os.path.exists(vDriftHistos): raise RuntimeError,'Could not produce %s' % vDriftHistos

	    # Write vDrift DB
	    if not os.path.exists(vDrift_segment_db): self.runVDriftSegmentWriter(run,vDriftHistos,config)
	    if not os.path.exists(vDrift_segment_db): raise RuntimeError,'Could not produce %s' % vDrift_segment_db

	elif mode == 'meantimer':
	    vDriftTMaxHistos = os.path.abspath(result_dir + '/' + 'DTTMaxHistos_' + run + '.root')
	    vDrift_meantimer_db = os.path.abspath(result_dir + '/' + 'vDrift_meantimer_' + run + '.db')

	    if not execute:
		print "Writing configuration files.."
		self.runVDriftMeanTimerCalib(run,runselection,trial,'VDriftTMaxHistos',vDriftTMaxHistos,config,False)
		self.runVDriftMeanTimerWriter(run,vDriftTMaxHistos,config,False)

		sys.exit(0)

	    # Produce t_max histos
	    if not os.path.exists(vDriftTMaxHistos):
		self.runVDriftMeanTimerCalib(run,runselection,trial,'VDriftTMaxHistos',vDriftTMaxHistos,config)
	    if not os.path.exists(vDriftTMaxHistos): raise RuntimeError,'Could not produce %s' % vDriftTMaxHistos

	    # Write vDrift DB
	    if not os.path.exists(vDrift_meantimer_db): self.runVDriftMeanTimerWriter(run,vDriftTMaxHistos,config)
	    if not os.path.exists(vDrift_meantimer_db): raise RuntimeError,'Could not produce %s' % vDrift_meantimer_db

	return 0        

    ############################################################ 
    # noise workflow
    ############################################################
    def runNoiseWorkflow(self,mode,run,config,execute=True):
	print "Processing noise calibration"

	trial = config.trial
	runselection = config.runselection
	result_dir = config.result_dir
	result_file = os.path.abspath(result_dir + '/' + 'dtNoiseCalib_' + run + '.root')
	noise_db = os.path.abspath(result_dir + '/' + 'noise_' + run + '.db')
	noise_txt = os.path.abspath(result_dir + '/' + 'noise_' + run + '.txt')
       
	datasetstr = getDatasetStr(config.datasetpath)
	#config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/NoiseCalibration/' + label + '/' + 'v' + str(trial)
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/NoiseCalibration/' + 'v' + str(trial)

	task_dir = config.base_dir + '/NoiseCalib'
	dtNoiseCalibration = DTNoiseCalibration(run=run,
						dir=task_dir,
						config=config)
	if not execute:
	    dtNoiseCalibration.writeCfg()
	    sys.exit(0)
	else:
	    dtNoiseCalibration.writeCfg()
	    project_noise = dtNoiseCalibration.run()

	    print "Sent calibration jobs with project",project_noise
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabNoiseCalibration = CrabWatch(project_noise)
	    crabNoiseCalibration.setThreshold(config.jobsFinishedThreshold)
	    crabNoiseCalibration.start()
	    crabNoiseCalibration.join()     

	    if config.stageOutLocal:
		crab_output_dir = project_noise + "/res"
		retcode = copyFilesLocal(crab_output_dir,result_dir,'dtNoiseCalib')
		retcode = copyFilesLocal(crab_output_dir,result_dir,'noise')
	    elif config.stageOutCAF:
		castor_dir = config.castorpath + "/" + config.userdircaf
		retcode = copyFilesFromCastor(castor_dir,result_dir,'dtNoiseCalib')
		retcode = copyFilesFromCastor(castor_dir,result_dir,'noise')

	return 0

    ############################################################ 
    # t0 workflow
    ############################################################
    def runT0Workflow(self,mode,run,config,execute=True):

	return 0

    ############################################################ 
    # Validation workflow
    ############################################################
    def runValidationWorkflow(self,mode,run,config,execute=True):
	print "Processing DQM validation"
	trial = config.trial
	runselection = config.runselection
	result_dir = config.result_dir
	datasetstr = getDatasetStr(config.datasetpath)
	dirLabel = 'DQM'
	config.userdircaf = 'DTCalibration/' + datasetstr + '/Run' + str(run) + '/DQMValidation/' + dirLabel + '/' + 'v' + str(trial)

	task_dir = config.base_dir + '/' + dirLabel
	dtDQMValid = DTDQMValidation(run=run,
				     dir=task_dir,
				     config=config)
	if not execute:
	    dtDQMValid.writeCfg()
	    self.runDQMMerge(run,'',config,False)
	    self.runDQMHarvesting(run,'',config,False)

	    sys.exit(0)
	else:
	    dtDQMValid.writeCfg()
	    project_valid = dtDQMValid.run()

	    print "Sent validation jobs with project",project_valid
	    print "%.0f%% of jobs are required to finish" % config.jobsFinishedThreshold

	    crabValid = CrabWatch(project_valid)
	    crabValid.setThreshold(config.jobsFinishedThreshold)
	    crabValid.start()
	    crabValid.join()

	    # Produce DQM output 
	    dqm_output_dir = self.dqmOutputDir('DQMValidation',dirLabel,config)
	    self.runDQMMerge(run,dqm_output_dir,config)
	    # Run harvesting from merged DQM file 
	    dqm_merge_dir = os.path.abspath(result_dir)
	    self.runDQMHarvesting(run,dqm_merge_dir,config)

	return 0

    ############################################################ 
    # Analysis workflow
    ############################################################ 
    def runAnalysisWorkflow(self,mode,run,config,execute=True):
	print "Processing analysis workflow"
	trial = config.trial
	runselection = config.runselection
	result_dir = config.result_dir
	if mode == 'residuals':
	    residualsFile = os.path.abspath(result_dir + '/' + 'DTResiduals_' + run + '.root')

	    if not execute:
		print "Writing configuration files.."
		self.runAnalysisResiduals(run,runselection,trial,'Residuals',residualsFile,config,False)

		sys.exit(0)

	    # Produce residuals
	    if not os.path.exists(residualsFile):
		self.runAnalysisResiduals(run,runselection,trial,'Residuals',residualsFile,config) 
	    if not os.path.exists(residualsFile): raise RuntimeError,'Could not produce %s' % residualsFile

	return 0

    ############################################################ 
    # DB Validation workflow
    ############################################################
    def runDBValidationWorkflow(self,mode,run,inputFiles,config,execute=True):
	print "Processing DB validation"

        dtDBValidation = None
        if mode == 't0DB':
	    dtDBValidation = DTT0DBValidation(run=run,
		    			      dir=config.run_dir,
					      input_files=inputFiles,
					      output_dir=config.result_dir,
					      config=config)
            #dtDBValidation.writeCfg()

	if execute:
	    dtDBValidation.run()

	    print "Finished processing:"
	    for pset in dtDBValidation.configs: print "--->",pset

        return 0
