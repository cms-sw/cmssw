import subprocess
import sys
import os
def submit(dataset, run):
	cfg=""
	with open("crabConfig.py","r") as f:
		cfg=f.readlines()
		cfg[0]="runNumber = "+str(run)+'\n'
		cfg[1]="dataset = "+'\"'+str(dataset)+'\"\n'
		
	with open("crabConfig.py","w") as f:
		f.writelines(cfg)
			
		
	#proc_id=subprocess.Popen(["crab", "submit", "-c", "crabConfig"+str(run)+".py"],start_new_session=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	subprocess.run(["crab", "submit", "-c", "crabConfig.py"],stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	return proc_id

proc_id=[]
runs=[]
submit_child=os.fork()
if(submit_child==0):
	if sys.argv[2]=="range":
		for run in range(int(sys.argv[3]),int(sys.argv[4])+1):
			runs.append(run)
			proc_id.append(submit(sys.argv[1], run))
	else:
		for run in sys.argv[2:]:
			runs.append(run)
			proc_id.append(submit(sys.argv[1], run))
		
else:
	print("submission process detached")
		
