###
# Sumbmit scripts for condor/dag jobs
###
skimSubTemplate="""
Executable = {base}/src/Alignment/APEEstimation/test/apeMeasurement/skimScript.tcsh
Universe = vanilla
Output = {workingArea}/skim_{name}_out.txt
Error  = {workingArea}/skim_{name}_err.txt
Log  = {workingArea}/skim_{name}_condor.log
request_memory = 2000M
request_disk = 2000M
batch_name = skim_{name}
arguments = "{base} '{args}' {target} {name}"
+JobFlavour = "tomorrow"
+AccountingGroup = "group_u_CMS.CAF.ALCA"
Queue
"""

prepSubTemplate="""
Executable = {base}/src/Alignment/APEEstimation/test/apeMeasurement/prepScript.tcsh
Universe = vanilla
Output = {workingArea}/{measName}/prep_out.txt
Error  = {workingArea}/{measName}/prep_err.txt
Log  = {workingArea}/{measName}/prep_condor.log
request_memory = 2000M
request_disk = 100M
batch_name = prep_{measName}
arguments = "{base} {workingArea} {globalTag} {measName} {isCosmics} {maxIterations} {baselineName} {dataDir} {fileName} {maxEvents} {isBaseline}"
+JobFlavour = "espresso"
+AccountingGroup = "group_u_CMS.CAF.ALCA"
Queue
"""


refitterSubTemplate="""
Executable = {base}/src/Alignment/APEEstimation/test/apeMeasurement/refittingScript.tcsh
Universe = vanilla
Output = {workingArea}/{measName}/refit_out_iter{iteration}_{fileNumber}.txt
Error  = {workingArea}/{measName}/refit_err_iter{iteration}_{fileNumber}.txt
Log  = {workingArea}/{measName}/refit_log_iter{iteration}_{fileNumber}.txt
request_memory = 2000M
request_disk = 500M
batch_name = refitting_{measName}_iter{iteration}_file{fileNumber}
arguments = "{base} {fileLocation} {workingArea} {globalTag} {measName} {fileNumber} {iteration} {lastIter} {isCosmics} {maxEvents}"
+JobFlavour = "longlunch"
+AccountingGroup = "group_u_CMS.CAF.ALCA" 
Queue 
"""

iterationSubTemplate="""
Executable = {base}/src/Alignment/APEEstimation/test/apeMeasurement/finishIterationScript.tcsh
Universe = vanilla
Output = {workingArea}/{measName}/iteration_out_iter{iteration}.txt
Error  = {workingArea}/{measName}/iteration_err_iter{iteration}.txt
Log  = {workingArea}/{measName}/iteration_log_iter{iteration}.txt
request_memory = 2000M
request_disk = 500M
batch_name = iteration_{measName}_iter{iteration}
arguments = "{base} {workingArea} {measName} {numFiles} {iteration} {isBaseline} {baselineName}"
+JobFlavour = "espresso"
+AccountingGroup = "group_u_CMS.CAF.ALCA" 
Queue 
"""

-- dummy change --
