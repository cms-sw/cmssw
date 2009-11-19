#include "DQM/Physics/plugins/VertexChecker.h"

VertexChecker::VertexChecker(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  dqmStore_ = edm::Service<DQMStore>().operator->();
  outputFileName_ = iConfig.getParameter<std::string>("outputFileName");
  vertex_        =   iConfig.getParameter<edm::InputTag>( "vertexName" );
  saveDQMMEs_    =    iConfig.getParameter<bool>("saveDQMMEs");
}

VertexChecker::~VertexChecker()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   //delete dqmStore_;
}

void
VertexChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;

  //Here you handle the collection you want to access
  edm::Handle< edm::View<reco::Vertex> > vertex;
  iEvent.getByLabel(vertex_, vertex);
  
  //Check if branch are available
  if (!vertex.isValid()) throw cms::Exception("ProductNotFound") <<"Primary vertex collection not found"<<std::endl;
  
  
  double pt2_vtx =0;
  double sumpt2_vtx =0;
  double ptAssTrk = -100;
  int prim_vtxidx = -2;
  double sumpt =0;
  double sumpt_prim =0;
  double pt2_primvtx =  0;
  double sumpt2_primvtx =0;
  //  Primary Vertex infos;
  if(vertex->size() > 0){
    histocontainer_["VtxNumber"]->Fill(vertex->size());
    for (unsigned int j=0; j< vertex->size(); j++){
      
      histocontainer_["Vrt_x"]->Fill((*vertex)[j].x());
      histocontainer_["Vrt_y"]->Fill((*vertex)[j].y());
      histocontainer_["Vrt_z"]->Fill((*vertex)[j].z());
      histocontainer_["Vrt_xy"]->Fill((*vertex)[j].x(),(*vertex)[j].y() );
      histocontainer_["VrtTrkNumber"]->Fill((*vertex)[j].tracksSize());
      reco::Vertex::trackRef_iterator tr ;  
      for ( tr = (*vertex)[j].tracks_begin(); tr !=(*vertex)[j].tracks_end(); tr++){
	pt2_vtx =  (*tr)->pt() * (*tr)->pt() ;
	sumpt2_vtx += pt2_vtx;
	sumpt += (*tr)->pt();
	histocontainer_["VrtTrkPt"]->Fill((*tr)->pt());
      }
      if( sumpt2_vtx > ptAssTrk ){ //defines as "primary" the vertex associated with the highest sum of pt2 of the tracks in the event
	ptAssTrk = sumpt2_vtx;
	prim_vtxidx =j; 
      }
    }
    histocontainer_["PrimaryVrt_x"]->Fill((*vertex)[ prim_vtxidx].x());
    histocontainer_["PrimaryVrt_y"]->Fill((*vertex)[ prim_vtxidx].y());
    histocontainer_["PrimaryVrt_z"]->Fill((*vertex)[ prim_vtxidx].z());   
    histocontainer_["VrtTrkSumPt"]->Fill(sumpt);
    histocontainer_["PrimaryVrt_xy"]->Fill((*vertex)[ prim_vtxidx].x(), (*vertex)[ prim_vtxidx].y());
    histocontainer_["PrimaryVrtTrkNumber"]->Fill((*vertex)[prim_vtxidx].tracksSize());
    reco::Vertex::trackRef_iterator trp ;  
    for ( trp = (*vertex)[ prim_vtxidx].tracks_begin(); trp !=(*vertex)[ prim_vtxidx].tracks_end(); trp++){
      sumpt_prim += (*trp)->pt();
      pt2_primvtx =  (*trp)->pt() * (*trp)->pt() ;
      sumpt2_primvtx += pt2_primvtx;
    }
    histocontainer_["PrimVrtTrkSumPt"]->Fill(sumpt_prim);
    histocontainer_["PrimVrtTrkSumPt2"]->Fill(sumpt2_primvtx);
  }   
}

void 
VertexChecker::beginJob(const edm::EventSetup&)
{
  dqmStore_->setCurrentFolder( "Vertex" );
  
  histocontainer_["VtxNumber"] = dqmStore_->book1D("VtxNumber" ,"Number of vertices ",5,0, 5);
  histocontainer_["VtxNumber"]->setAxisTitle("Nof vertices",1);
  histocontainer_["Vrt_x"]= dqmStore_->book1D("Vtx_x" ,"vertices x ",50,-1, 1);
  histocontainer_["Vrt_x"]->setAxisTitle("Vertices: X",1);
  histocontainer_["Vrt_y"]= dqmStore_->book1D("Vtx_y" ,"vertices y ",50,-1, 1);
  histocontainer_["Vrt_y"]->setAxisTitle("Vertices: Y",1);
  histocontainer_["Vrt_z"]= dqmStore_->book1D("Vtx_z" ,"vertices z ",100,-10, 10);
  histocontainer_["Vrt_z"]->setAxisTitle("Vertices: Z",1);
  histocontainer_["Vrt_xy"]= dqmStore_->book2D("Vtx_xy" ,"vertices x vs y ",50,-1, 1, 50, -1, 1);
  histocontainer_["Vrt_xy"]->setAxisTitle("Vertices: XY",1);
  histocontainer_["PrimaryVrt_x"]= dqmStore_->book1D("PrimaryVtx_x" ,"Primary vertex x ",50,-1, 1);
  histocontainer_["PrimaryVrt_x"]->setAxisTitle("Primary Vertex: X",1);
  histocontainer_["PrimaryVrt_y"]= dqmStore_->book1D("PrimaryVtx_y" ,"Primary vertex y ",50,-1, 1);
  histocontainer_["PrimaryVrt_y"]->setAxisTitle("Primary Vertex: Y",1);
  histocontainer_["PrimaryVrt_z"]= dqmStore_->book1D("PrimaryVtx_z" ,"Primary vertex z ",100,-10, 10);
  histocontainer_["PrimaryVrt_z"]->setAxisTitle("Primary Vertex: Z",1);
  histocontainer_["PrimaryVrt_xy"] =  dqmStore_->book2D("PrimaryVtx_xy" ,"Primary vertex x vs y ",50,-1, 1, 50, -1, 1);
  histocontainer_["PrimaryVrt_xy"]->setAxisTitle("Primary Vertex: XY",1);
  histocontainer_["PrimaryVrtTrkNumber"]= dqmStore_->book1D("PrimaryVrtTrkNumber","Number of tracks associated to primary vertex",10,0, 10);
  histocontainer_["PrimaryVrtTrkNumber"]->setAxisTitle("Nof tracks associated to PrimVertex",1);
  histocontainer_["VrtTrkNumber"] = dqmStore_->book1D("VrtTrkNumber" ,"Number of tracks associated to vertices ",10,0, 10);
  histocontainer_["VrtTrkNumber"]->setAxisTitle("Nof tracks associated to Vertex",1);
  histocontainer_["VrtTrkPt"] = dqmStore_->book1D("VrtTrkPt" ,"track associated to vertices Pt ",100,0, 50);
  histocontainer_["VrtTrkPt"]->setAxisTitle("Pt of tracks associated to Vertex",1);
  histocontainer_["VrtTrkSumPt"]= dqmStore_->book1D("VrtTrkSumPt" ," Sum of the pt of the tracks associated to vertices",100,0, 300);
  histocontainer_["VrtTrkSumPt"]->setAxisTitle("SumPt of tracks associated to Vertex",1);
  histocontainer_["PrimVrtTrkSumPt"]= dqmStore_->book1D("PrimVrtTrkSumPt" ," Sum of the pt of the tracks associated to Primary vertex",50,0, 300);
  histocontainer_["PrimVrtTrkSumPt"]->setAxisTitle("SumPt of tracks associated to PrimVertex",1);
  histocontainer_["PrimVrtTrkSumPt2"]= dqmStore_->book1D("PrimVrtTrkSumPt2" ,"Sum of the pt^2 of the tracks associated to Primary vertex",50,0, 10000);
  histocontainer_["PrimVrtTrkSumPt2"]->setAxisTitle("SumPt2 of tracks associated to PrimVertex",1);
}

void 
VertexChecker::endJob() 
{
  //use LogError to summarise the error that happen in the execution (by example from warning) (ex: Nof where we cannot access such variable)
  //edm::LogError  ("SummaryError") << "My error message \n";    // or  edm::LogProblem  (not formated)
  //use LogInfo to summarise information (ex: pourcentage of events matched ...)
  
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " --     Report from Vertex Checker       -- ";
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << "Mean Number of Vertices  = " << histocontainer_["VtxNumber"]->getMean() ;  
  edm::LogVerbatim ("MainResults") << "Vertices Mean x = " << histocontainer_["Vrt_x"]->getMean() ;  
  edm::LogVerbatim ("MainResults") << "Vertices Mean y = " << histocontainer_["Vrt_y"]->getMean() ;  
  edm::LogVerbatim ("MainResults") << "Vertices Mean z = " << histocontainer_["Vrt_z"]->getMean() ;  
  edm::LogVerbatim ("MainResults") << " -------------------------------------------";
  edm::LogVerbatim ("MainResults") << " -----    Primary Vertex Info  -------------";
  edm::LogVerbatim ("MainResults") << "Prim Vertex Mean x = " << histocontainer_["PrimaryVrt_x"]->getMean() ;  
  edm::LogVerbatim ("MainResults") << "Prim Vertex Mean y = " << histocontainer_["PrimaryVrt_y"]->getMean() ;  
  edm::LogVerbatim ("MainResults") << "Prim Vertex Mean z = " << histocontainer_["PrimaryVrt_z"]->getMean() ;  
  
  if(saveDQMMEs_)
    dqmStore_->save(outputFileName_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(VertexChecker);
