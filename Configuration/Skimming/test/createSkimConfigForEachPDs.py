import os

from Configuration.Skimming.autoSkim import autoSkim

for k in autoSkim:
   #    print k
   #print ('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012A-PromptReco-v1/RECO and run=190705\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
   os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012A-PromptReco-v1/RECO and run=190705\" -n 1000 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
