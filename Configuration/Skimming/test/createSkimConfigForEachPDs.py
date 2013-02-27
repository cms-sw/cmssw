import os
import time

localtime = time.localtime(time.time())

#print "Local current time :", localtime

from Configuration.Skimming.autoSkim import autoSkim

tier1dirname = 'tier1_%sy_%sm_%sd_%sh_%sm_%ss' %(localtime[0],localtime[1],localtime[2],localtime[3],localtime[4],localtime[5])
testdirname = 'test_%sy_%sm_%sd_%sh_%sm_%ss' %(localtime[0],localtime[1],localtime[2],localtime[3],localtime[4],localtime[5])
os.system( 'mkdir %s' %(tier1dirname) )
os.system( 'mkdir %s' %(testdirname) )

for k in autoSkim:
   print k

   if "Cosmics" in k:
      ### 5E33 menu
      #os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012A-PromptReco-v1/RECO and run=191277\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py --scenario=cosmics' %(k,k,k))
      ### 7E33 menu
      #os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012B-PromptReco-v1/RECO and run=193928\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py --scenario=cosmics' %(k,k,k))
      os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012B-PromptReco-v1/RECO and run=194050 and lumi=500\" -n 1000 --conditions auto:com10 --python_filename=skim_%s.py --scenario=cosmics' %(k,k,k))

   else:
      ### 5E33 menu
      #print ('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012A-PromptReco-v1/RECO and run=190705\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
      #os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012A-PromptReco-v1/RECO and run=190705\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
      #os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012A-PromptReco-v1/RECO and run=191277\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
      ### 7E33 menu
      #os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012B-PromptReco-v1/RECO and run=193928\" -n 100 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
      os.system('cmsDriver.py skims -s SKIM:@%s --data --no_exec --dbs \"find file,file.parent where dataset=/%s/Run2012B-PromptReco-v1/RECO and run=194050 and lumi=500\" -n 1000 --conditions auto:com10 --python_filename=skim_%s.py' %(k,k,k))
      
   os.system('mkdir -p %s/%s' %(testdirname,k))

   ######################################################
   # uncomment below if you want to get a summary from your test
   ######################################################
   os.system('echo \"process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )\" >> skim_%s.py' %(k))

   os.system('cp skim_%s.py %s/%s' %(k,testdirname,k))
   os.system('mv skim_%s.py %s' %(k,tier1dirname))


   ######################################################
   # uncomment below if you want to run a test on all the skims
   ######################################################
   #os.system('cd %s/%s ; cmsRun skim_%s.py > skim_%s.txt 2>&1' %(testdirname,k,k,k) )
   # use the one below in case RAW not available on eos
   #print ('cd %s/%s ; sed -i \"s/\/store\/data\/Run2012B/rfio\:\/castor\/cern.ch\/cms\/store\/data\/Run2012B/g\" skim_%s.py ; cmsRun skim_%s.py > skim_%s.txt 2>&1' %(testdirname,k,k,k,k) )
   #os.system('cd %s/%s ; sed -i \"s/\/store\/data\/Run2012B/rfio\:\/castor\/cern.ch\/cms\/store\/data\/Run2012B/g\" skim_%s.py ; cmsRun skim_%s.py > skim_%s.txt 2>&1 &' %(testdirname,k,k,k,k) )
