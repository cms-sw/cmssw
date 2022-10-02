import subprocess
import sys
def submit(dataset, run):
	cfg=""
	with open("crabConfig.py","r") as f:
		cfg=f.readlines()
		cfg[0]="runNumber = "+str(run)+'\n'
		cfg[1]="dataset = "+'\"'+str(dataset)+'\"\n'
		
	with open("crabConfig.py","w") as f:
		f.writelines(cfg)
			
		
	subprocess.run(["crab", "submit", "-c", "crabConfig.py"])
	
submit_child=os.fork()	
if submit_child==0:	
	if sys.argv[2]=="range":
		for run in range(int(sys.argv[3]),int(sys.argv[4])+1):
			submit(sys.argv[1], run)
	else:
		for run in sys.argv[2:]:
			submit(sys.argv[1], run)
else:
	print("submission process detached")
		
