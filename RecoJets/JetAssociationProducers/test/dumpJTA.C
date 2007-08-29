{
   #include "DataFormats/FWLite/interface/Handle.h"

   TFile file("/scratch/ratnikov/data/RelVal152BJets50-120_17X_jettrackassociation.root");

   fwlite::Event ev(&file);

   for( ev.toBegin();
          ! ev.atEnd();
          ++ev) {
     std::cout << "=================== event begin ====================" << std::endl;
     //       fwlite::Handle<reco::JetToTracksAssociation::Container> > objs;
       fwlite::Handle<std::vector <std::pair <edm::RefToBase<reco::Jet>, reco::TrackRefVector> > > objs;
       objs.getByLabel(ev,"ic5JetToTracksAssociator");
      //now can access data
      std::cout <<" size "<<objs->size()<<std::endl;
      std::vector<edm::RefToBase<reco::Jet> > jets = reco::JetToTracksAssociation::allJets (*objs);
      std::cout << "Total jets: " << jets.size() << std::endl;
      for (unsigned j = 0; j < jets.size(); ++j) {
	std::cout << "Jet #" << j << std::endl
		  << jets[j]->print() << std::endl;
	reco::TrackRefVector tracks = reco::JetToTracksAssociation::getValue (*objs, jets[j]);
	std::cout << "Associated traks: " << tracks.size() << std::endl;
	for (unsigned t = 0; t < tracks.size(); ++t) {
 	  std::cout << "track p/pt/eta/phi: " << (tracks[t])->p() << '/' << (tracks[t])->pt() << '/' 
  		    << (tracks[t])->eta() << '/' << (tracks[t])->phi() << std::endl;
	}
      }
      std::cout << "=================== event end ====================" << std::endl;
   }
}
