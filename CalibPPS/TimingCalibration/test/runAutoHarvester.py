from pathlib import Path
import sys
import subprocess
def submit(path, run):
	cfg=""
	with open("DiamondCalibrationHarvester_cfg.py","r") as f:
		cfg=f.readlines()
		cfg[0]="run = "+str(run)+'\n'
		cfg[1]="input_file="+str(path_list)+'\n'
		
	with open("DiamondCalibrationHarvester_cfg.py","w") as f:
		f.writelines(cfg)
			
		
	subprocess.run(["cmsRun", "DiamondCalibrationHarvester_cfg.py"])
	
	
path_list=[]
last_run=0;
run=0
for path in Path(sys.argv[1]).rglob(sys.argv[2]):
    path=str(path)
    start=path.rfind(sys.argv[3])
    if(start==-1):
    	continue
    start=start+len(sys.argv[3])
    end=path.find("/",start)

    run=sys.argv[4]+path[start:end]
    print(str(path), run)
    if(last_run==0 or run==last_run): 
    	path_list.append("file:"+str(path))
    else:
    	submit(path_list,last_run) 
    	path_list=[]
    	path_list.append("file:"+str(path))
    last_run=run

if len(path_list)!=0:
	submit(path_list,last_run) 
  
