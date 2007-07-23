///////////////////////////////////////////////////////////////////////////////
//
// Author M. De Mattia demattia@pd.infn.it
// date 9/5/2007
//
// Simple FWLite macro example to use the AnalyzedObjects (Track and Cluster)
//
// Note: cannot use the typedefs before having done SetBranchAddess for the
// corresponding branch.
//
// Usage: change the name of the files to use in the chain,
// .L FWLite.C
// FWLite.C
//
///////////////////////////////////////////////////////////////////////////////

// The number indicates the starting array lenght
TObjArray Hlist(0);

void BookHistos () {

//  Hlist.Add( new TH1F( "Theta", "Theta", 100, -180, 180 );

}

void FWLite () {

  gROOT->Reset();

  gSystem->Load("libFWCoreFWLite.so"); 
  AutoLibraryLoader::enable();
  using namespace std;
  using namespace edm;
  using namespace anaobj;

  TChain events("Events");

  events.Add("/data/demattia/TIFAnaObjProducer/Run_6509_v5.root");

  //set the buffers for the branches
  // AnalyzedClusters
  std::vector<AnalyzedCluster> v_anaclu;
  TBranch* anaclu_B;
  events.SetBranchAddress("anaobjAnalyzedClusters_modAnaObjProducer__TIFAnaObjProducer.obj",&v_anaclu,&anaclu_B);
  // AnalyzedTracks
  std::vector<AnalyzedTrack> v_anatk;
  TBranch* anatk_B;
  events.SetBranchAddress("anaobjAnalyzedTracks_modAnaObjProducer__TIFAnaObjProducer.obj",&v_anatk,&anatk_B);

// Do not define them, otherwise it will not work
// ----------------------------------------------
//  anaobj::AnalyzedClusterCollection v_anaclu;
//  anaobj::AnalyzedClusterRef anacluRef;
//  anaobj::AnalyzedClusterRefVector anacluRefVec;

//  anaobj::AnalyzedTrackCollection v_anatk;
//  anaobj::AnalyzedTrackRef anatkRef;
//  anaobj::AnalyzedTrackRefVector anatkRefVec;
// ----------------------------------------------

//  anacluRefVec.size();

  //loop over the events
  for( unsigned int index = 0;
       index < events.GetEntries();
       ++index) {
    //need to call SetAddress since TBranch's change for each file read
    anaclu_B->SetAddress(&v_anaclu);
    anaclu_B->GetEntry(index);
    anatk_B->SetAddress(&v_anatk);
    anatk_B->GetEntry(index);
    events.GetEntry(index,0);

    //now can access data

    std::cout << "Event = " << index << std::endl;

    std::cout <<"Number of clusters = "<<v_anaclu.size()<<std::endl;
    std::cout <<"Number of tracks = "<<v_anatk.size()<<std::endl;

    // The iterator does not work, check in classes.h and classes_def.xml for additional wrappers to be defined
//    anaobj::AnalyzedTrackCollection::iterator anatk_iterator;
//    for ( anatk_iterator = v_anatk.begin(); anatk_iterator != v_anatk.end(); ++anatk_iterator ) {
//      std::cout << "track clusters = " << anatk_iterator->GetClusterRefVec().size() << std::endl;
//      std::cout << "rechits number = " << anatk_iterator->hitspertrack << std::endl;
//    }

// CANNOT USE THE REFs IT RETURNS ALWAYS THE POINTERS TO THE FIRST COLLECTION IN THE ROOT FILE (it works in the EDAnalyzer)

    // Loop on the tracks
    for ( int anatk_iter = 0; anatk_iter < v_anatk.size(); ++anatk_iter ) {
      // Take the Track
      AnalyzedTrack Track( v_anatk[anatk_iter] );

      // Can use the track here to access the variables
      // to see the data members look in /AnalysisExamples/AnalyzedObjects/interface/AnalyzedTrack.h

      // Take the vector with the indeces to the clusters associated with this track
      std::vector<int> v_clu_id( Track.clu_id );
      int clu_id_size = v_clu_id.size();

      // Now loop on the vector of clusters belonging to this track
      for (int cluster_index=0; cluster_index < clu_id_size; ++cluster_index) {
	int cluster_id = v_clu_id[cluster_index];
	// Take the cluster
        AnalyzedCluster Cluster( v_anaclu[cluster_id] );

        // Use the cluster to access cluster related variables
	// to see the data members look in /AnalysisExamples/AnalyzedObjects/interface/AnalyzedCluster.h
	std::cout << "thickness for layer " << Cluster.layer
		  << " of detector "<< Cluster.type
		  << " is : " << Cluster.thickness << std::endl;

	// Use the cluster to access track related variables
        // These variables are stored in maps whose key is the id of the track
        if ( Track.tk_id-1 == anatk_iter ) {
	  float angle3D = Cluster.angle3D.find( Track.tk_id )->second;
	  std::cout << "angle3D( "<<Track.tk_id<<" ) = " << angle3D << std::endl;
	}
	else {
	  std::cout << "The order is changed" << std::endl;
	}
      }
    }


    // Another example this time looping on clusters
    // ---------------------------------------------

    // Loop on all the analyzed clusters
    for ( int anaclu_iter = 0; anaclu_iter < v_anaclu.size(); ++anaclu_iter ) {

      // Take the vector with the track index
      std::vector<int> v_tk_id( v_anaclu[anaclu_iter].tk_id );
      int tk_id_size = v_tk_id.size();

      std::map<int, float> angle_map( v_anaclu[anaclu_iter].angle );
      std::map<int, int>::const_iterator tk_angle_map_iter;

      for ( int id=0; id < tk_id_size; ++id ) {

        int tk_id = v_tk_id[id];

	std::map<int, float>::iterator angle_map_iter = angle_map.find(tk_id);
	float angle_val = angle_map_iter->second;
//	std::cout << "angle_val(tk_id = "<<tk_id<<") = " << angle_val << std::endl;
      }
    }

  }
};
