
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
  string cut = "@particles_.size()==1 && abs(particles_[0].eta)<";   
  // string cut = "abs(particles_.eta)<";   
  cut += cetamax;
 
  string cutseen = cut;
  
  switch(clustertype) {
  case 0:
    cutseen += " && sqrt((clusters_.eta-particles_[0].eta)^2 + (clusters_.phi-particles_[0].phi)^2) < ";
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
  h->Sumw2();
  ref->Sumw2();
  h->Divide(ref);

  return h;
}  

void Impurity( int clustertype, 
	       float drmax=0.1,
	       float etamax=1,
	       const char* hname="impurity") {
  
  string eflowname = hname; eflowname += "_eflow";
  string islandname = hname; islandname += "_island";
  string refname = hname; refname += "_ref";
  
  int nbinse = 100;
  float mine = 0;
  float maxe = 200;

  TH2D* heflow=new TH2D(eflowname.c_str(), eflowname.c_str(), 
			nbinse, mine, maxe, 10,0,10);
  TH2D* hisland = (TH2D*) heflow->Clone(islandname.c_str());
  TH1D* href = new TH1D(refname.c_str(), refname.c_str(), 
			nbinse, mine, maxe);
  
  EventColin* event = new EventColin();
  chain->SetBranchAddress("event",&event);
  
//   vector<int> truc;
//   cout<<truc.size()<<endl;

  for(unsigned i=0; i<chain->GetEntries(); i++) {

    if( (i%1000) == 0) cout<<i<<endl;

    chain->GetEntry(i);
    // count the number of clusters close to the particle
    
    const std::vector<EventColin::Particle>& 
      particles = event->particles(); 

    if(particles.size() != 1) {
      // cerr<<"number of particles != 1 - skip"<<endl;
      continue;
    }
    
    if( fabs(particles[0].eta) > etamax) continue;

    href->Fill(particles[0].e);

//     const vector<EventColin::Cluster>* clusters;
//     switch(clustertype) {
//     case 0:
//       clusters = &  event->clusters(); 
//       break;
//     case 1:
//       clusters = &  event->clustersIsland(); 
//       break;
//     default:
//       cerr<<"clustertype "<<clustertype<<" not recognized"<<endl;
//       return;
//     }

//     const vector<EventColin::Cluster>& cref = *clusters;
    
//     const std::vector<EventColin::Cluster>& 
//       clusters = event->clusters(); 

//     const std::vector<EventColin::Cluster>& 
//       clustersIsland = event->clustersIsland(); 


    const vector<EventColin::Cluster>& clustersEflow = event->clusters();
    int nclustersOkeflow = 0;
    for( unsigned jc = 0; jc<clustersEflow.size(); jc++ ) {
      double dphi = clustersEflow[jc].phi - particles[0].phi;
      double deta = clustersEflow[jc].eta - particles[0].eta;
      double dr = sqrt(dphi*dphi + deta*deta);
      
      if(dr<drmax) nclustersOkeflow++;
    }
    
    heflow->Fill( particles[0].e, nclustersOkeflow );
    
    const vector<EventColin::Cluster>& clustersIsland = event->clustersIsland();
    int nclustersOkisland = 0;
    for( unsigned jc = 0; jc<clustersIsland.size(); jc++ ) {
      double dphi = clustersIsland[jc].phi - particles[0].phi;
      double deta = clustersIsland[jc].eta - particles[0].eta;
      double dr = sqrt(dphi*dphi + deta*deta);

      if(dr<drmax) nclustersOkisland++;
    }
    
    hisland->Fill( particles[0].e, nclustersOkisland );  
  }

  return;
}


// TH1F* ImpurityRatio( const TH2F* h ) {

 
// }



TH1F* EffPlateau2( Chain* chain, 
		   int clustertype, 
		   const char* hname="effplateau", 
		   float drmax=0.1,
		   float etamax=1 ) {
  

  string var = "particles_.e";

  string shname = hname;
  char type[2];
  sprintf(type,"%d",clustertype);
  shname += type;

  TH1F* h=new TH1F(shname.c_str(), shname.c_str(), 50,0,5);

  string nameref = shname;
  nameref += "_ref";
  TH1F* href = (TH1F*) h->Clone( nameref.c_str() );
  
  string nameseen = shname;
  nameseen += "_seen";
  TH1F* hseen = (TH1F*) h->Clone( nameseen.c_str() );


  EventColin* event = new EventColin();
  chain->SetBranchAddress("event",&event);
  

  for(unsigned i=0; i<chain->GetEntries(); i++) {

    if( (i%1000) == 0) cout<<i<<endl;
    chain->GetEntry(i);
    // count the number of clusters close to the particle
    
    const std::vector<EventColin::Particle>& 
      particles = event->particles(); 

    if(particles.size() != 1) {
      // cerr<<"number of particles != 1 - skip"<<endl;
      continue;
    }
    
    if( fabs(particles[0].eta) > etamax) continue;
    href->Fill( particles[0].e );

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
    
    if(nclustersOk)
      hseen->Fill( particles[0].e );
  }

  h->Add(hseen);
  h->Divide(href);

  return h;
}
