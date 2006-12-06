
Chain* chain;

using namespace std;

void Init(const char* files="Out_std/out_b_singlegamma_pf_*root") { 
  gROOT->Macro("init.C");
  
  chain = new Chain("Eff",files);
}

void EffIneff2D(int clustertype, const char* cut= "") {
  
  
  string hname = "effineff1"; 

  string var;
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

TH1F* EffPlateau(Chain* chain, 
		 int clustertype, 
		 const char* hname="effplateau", 
		 float drmax=0.1,
		 float etamax=1 ) { 
  
  // if(!chain) Init();

  string var = "particles_.e";

  string shname = hname;
  char type[2];
  sprintf(type,"%d",clustertype);
  shname += type;

  char cdrmax[10];
  sprintf(cdrmax,"%f",drmax);

  char cetamax[10];
  sprintf(cetamax,"%f",etamax);
  string cut = "@particles_.size()==1 && abs(particles_.eta)<";   
  cut += cetamax;
 
  string cutseen = cut;
  
  switch(clustertype) {
  case 0:
    cutseen += " && sqrt((clusters_.eta-particles_.eta)^2 + (clusters_.phi-particles_.phi)^2) < ";
    cutseen += cdrmax;

    cout<<"eflow"<<endl;
    break;
  case 1:
    cutseen += " && sqrt((clustersIsland_.eta-particles_.eta)^2 + (clustersIsland_.phi-particles_.phi)^2) < ";
    cutseen += cdrmax;

    cout<<"island"<<endl;
    break;
  }

  TH1F* h=new TH1F(shname.c_str(), shname.c_str(), 50,0,5);

  string nameref = shname;
  nameref += "_ref";
  TH1F* ref = (TH1F*) h->Clone( nameref.c_str() );

  string nameseen = shname;
  nameseen += "_seen";
  TH1F* seen = (TH1F*) h->Clone( nameseen.c_str() );
  
  string varref = var;
  varref += ">>"; varref += nameref;

  cout<<varref<<" "<<cut<<endl;
  chain->Draw(varref.c_str(), cut.c_str(), "goff");

  string varseen = var;
  varseen += ">>"; varseen += nameseen;
  cout<<varseen<<" "<<cutseen<<endl;
  chain->Draw(varseen.c_str(), cutseen.c_str(),"goff" );
  
  h->Add(seen);
  h->Divide(ref);

  return h;
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
