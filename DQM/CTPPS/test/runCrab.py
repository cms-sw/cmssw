import subprocess
import sys
import os
def submit(dataset, run, out_files):
	cfg=""
	with open("crabConfig.py","r") as f:
		cfg=f.readlines()
		cfg[0]="runNumber = \'"+str(run)+'\'\n'
		cfg[1]="dataset = "+'\"'+str(dataset)+'\"\n'
		cfg[2]="out_files = "+str(out_files)+'\n'
		
	with open("crabConfig.py","w") as f:
		f.writelines(cfg)
			
		
	subprocess.run(["crab", "submit", "-c", "crabConfig.py"],stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)




out_files=[]
submit_child=os.fork()
run_range=''
if(submit_child==0):
	if sys.argv[2]=="range":
		for run in range(int(sys.argv[3]),int(sys.argv[4])+1):
			out_files.append('DQM_V0001_CTPPS_R000'+str(run)+'.root')
		run_range=sys.argv[3]+'-'+sys.argv[4]
	else:
		for run in sys.argv[2:]:
			out_files.append('DQM_V0001_CTPPS_R000'+str(run)+'.root')
		run_range=sys.argv[2]+'-'+sys.argv[-1]
		
	submit(sys.argv[1], run_range, out_files)
		
else:
	print("submission process detached")
		
