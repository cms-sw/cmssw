
Chain* chain;

using namespace std;

void Init(const char* files="Out_std/out_b_singlegamma_pf_*root") { 
  gROOT->Macro("init.C");
  
  chain = new Chain("Eff",files);
}

void EffIneff2D(int clustertype, const char* cut= "") {
  
  
  string var;
  string hname = "effineff1"; 

  char type[2];
  sprintf(type,"%d",clustertype);
  hname += type;

  switch(clustertype) {
  case 0:
    cout<<"eflow"<<endl;
    var = "@clusters_.size()";
    break;
  case 1:
    cout<<"island"<<endl;
    var = "@clustersIsland_.size()";
    break;
  }
  var += ":particles_.e>>";
  var += hname;


  TH2F* h=new TH2F( hname.c_str(), hname.c_str(), 200, 0, 200, 4,0,4);

  chain->Draw( var.c_str(), cut, "lego2");
}


TH2F* EffIneff2( int clustertype, 
		 float drmax=0.1,
		 float etamax=1,
		 const char* hname="effineff2") {
  
  TH2F* h=new TH2F(hname, hname, 200, 0, 200, 4,0,4);

  EventColin* event = new EventColin();
  chain->SetBranchAddress("event",&event);
  
  vector<int> truc;
  cout<<truc.size()<<endl;

  for(unsigned i=0; i<chain->GetEntries(); i++) {

    chain->GetEntry(i);
    // count the number of clusters close to the particle
    
    const std::vector<EventColin::Particle>& 
      particles = event->particles(); 

    if(particles.size() != 1) {
      cerr<<"number of particles != 1 - skip"<<endl;
      continue;
    }
    
    if( fabs(particles[0].eta) > etamax) continue;

    const vector<EventColin::Cluster>* clusters;
    switch(clustertype) {
    case 0:
      clusters = &  event->clusters(); 
      break;
    case 1:
      clusters = &  event->clustersIsland(); 
      break;
    default:
      cerr<<"clustertype "<<clustertype<<" not recognized"<<endl;
      return;
    }

    const vector<EventColin::Cluster>& cref = *clusters;
    
//     const std::vector<EventColin::Cluster>& 
//       clusters = event->clusters(); 

//     const std::vector<EventColin::Cluster>& 
//       clustersIsland = event->clustersIsland(); 


    int nclustersOk = 0;
    for( unsigned jc = 0; jc<cref.size(); jc++ ) {
      double dphi = cref[jc].phi - particles[0].phi;
      double deta = cref[jc].eta - particles[0].eta;
      double dr = sqrt(dphi*dphi + deta*deta);

      if(dr<drmax) nclustersOk++;
    }
    
    h->Fill( particles[0].e, nclustersOk );
  }

  return h;
}
