

#include <iostream>
#include "slhcevent.h"


int main(const int argc, const char** argv)
{

  using namespace std;
  if (argc<5) {
    cout << "\n\nparameters: signal.txt Nsig pileup.txt Npu\n\n" << endl;
    assert(0);
  }

  int nevents = atoi(argv[2]);
  ifstream in;
  in.open(argv[1]);
  int NPU = atoi(argv[4]);

  cerr<<"mixing "<<nevents<<" of signal with "<<NPU<<" of PU each\n";

  vector<bool> innerStack;
  vector<int> irphi;
  vector<int> iz;
  vector<int> iladder;
  vector<int> imodule;

  for (int i=0;i<nevents;i++){

    SLHCEvent ev(in);

    cerr <<"Process event: "<<i<<endl;

    ifstream inPU;
    inPU.open(argv[3]);

    for(int iPU = 0; iPU < NPU; ++iPU){ // loop over the PU overlays
      SLHCEvent evPU(inPU);

      cerr << "PU event:"<<evPU.eventnum()<<" Number of stubs="
	 <<evPU.nstubs()<<endl;
      cerr << "Main event:"<<ev.eventnum()<<" Number of stubs="
	   <<ev.nstubs()<<endl;
    

      SLHCEvent newev; // the new event
      //first, add the simtracks from the main event
      int nsimtracks = ev.nsimtracks();
      for(int ist=0; ist<nsimtracks; ++ist){
	newev.addL1SimTrack(ev.simtrack(ist).id(),
			    ev.simtrack(ist).type(),
			    ev.simtrack(ist).pt(),
			    ev.simtrack(ist).eta(),
			    ev.simtrack(ist).phi(),
			    ev.simtrack(ist).vx(),
			    ev.simtrack(ist).vy(),
			    ev.simtrack(ist).vz());
      }

      //now add stubs

      int jPU = 0;
      for (int j=0;j<ev.nstubs();j++){
	//	cerr<<"     ++ adding stubs "<<j<<" "<<jPU<<"\n";

	L1TStub stub=ev.stub(j);

	//see which stubs from PU event should preceed the current stub.
	bool insertPU = jPU < evPU.nstubs();
	while(insertPU){
	  L1TStub stubPU = evPU.stub(jPU);

	  insertPU = false;	
	  if(stub.layer() < 10) {//barrel
	    if(stubPU.layer() < stub.layer())
	      insertPU = true;
	    else if(stubPU.layer() == stub.layer())
	      if(stubPU.ladder() < stub.ladder())
	        insertPU = true;
	      else if(stubPU.ladder() == stub.ladder())
	        if(stubPU.module() < stub.module())
		  insertPU = true;
		else if(stubPU.module() == stub.module()){//readout order from module
		  if(stub.layer() < 4){//PS: z is enough
		    if(stubPU.z() > stub.z())
		      insertPU = true;
		  }
		  else{ //SS
		    if(stubPU.z() > stub.z()+2.5)
		      insertPU = true;
		    else if(stubPU.phi() > stub.phi())
		      insertPU = true;
		  }
		}
	  }// barrel block end
	  else{ //endcap
	    if(stubPU.layer() < 10)
	      insertPU = true;
	    else {
	      int sZ = stub.z() > 0 ? 1 : -1;
	      int sPUZ = stubPU.z() > 0 ? 1 : -1;
	      if(sPUZ < sZ)
		insertPU = true;
	      else 
		if(stubPU.module() < stub.module())
		  insertPU = true;
		else if(stubPU.module() == stub.module())
		  if(stubPU.layer() < stub.layer())
		    insertPU = true;
		  else if(stubPU.layer() == stub.layer())
		    if(stubPU.ladder() < stub.ladder())
		      insertPU = true;
		    else if(stubPU.ladder() == stub.ladder()){ //readout order from module
		      if(stub.layer() < 1010){//PS
			if(stubPU.r() > stub.r()+0.2)
			  insertPU = true;
			else if(stubPU.strip() < stub.strip())
			  insertPU = true;
		      }
		      else {//SS
			if(stubPU.r() > stub.r() + 2.5)
			  insertPU = true;
			else if(stubPU.strip() < stub.strip())
			  insertPU = true;
		      }
		    }
	    }
	  }//endcap block end
	  
	  if(insertPU){
	    newev.addStub(stubPU.layer(), stubPU.ladder(), stubPU.module(), stubPU.strip(), stubPU.pt(), stubPU.bend(),
			  stubPU.x(), stubPU.y(), stubPU.z(),
			  innerStack, irphi, iz, iladder, imodule);
	    ++jPU;
            if (jPU >= evPU.nstubs()) { 
              cerr << " --- ran out of PU stubs" << i <<" "<<jPU<< endl;
              insertPU = false;
            }
	  }
	}
	newev.addStub(stub.layer(), stub.ladder(), stub.module(), stub.strip(), stub.pt(), stub.bend(),
		      stub.x(), stub.y(), stub.z(),
		      innerStack, irphi, iz, iladder, imodule);

      }
      cerr<<"     ++ done adding main stubs "<<jPU<<"\n";
      while(jPU < evPU.nstubs()){
	L1TStub stubPU = evPU.stub(jPU);
	newev.addStub(stubPU.layer(), stubPU.ladder(), stubPU.module(), stubPU.strip(), stubPU.pt(), stubPU.bend(),
		      stubPU.x(), stubPU.y(), stubPU.z(),
		      innerStack, irphi, iz, iladder, imodule);
	++jPU;
      }
      cerr<<"     ++ done adding stubs "<<jPU<<"\n";

      //output the new event
      newev.write(cout);

    } // end loop over the PU overlay

    inPU.close();

  }
} 
