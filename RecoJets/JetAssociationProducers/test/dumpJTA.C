{
   #include "DataFormats/FWLite/interface/Handle.h"

   TFile file("/scratch/ratnikov/data/RelVal152BJets50-120_17X_jettrackassociation.root");

   fwlite::Event ev(&file);

   for( ev.toBegin();
          ! ev.atEnd();
          ++ev) {
     std::cout << "=================== event begin ====================" << std::endl;
     fwlite::Handle<reco::CaloJetCollection> jets;
     jets.getByLabel(ev,"iterativeCone5CaloJets");
     fwlite::Handle<std::vector <std::pair <edm::RefToBase<reco::Jet>, reco::TrackRefVector> > > jet2tracksVX;
     jet2tracksVX.getByLabel(ev,"ic5JetToTracksAssociator");
     fwlite::Handle<std::vector <std::pair <edm::RefToBase<reco::Jet>, reco::TrackRefVector> > > jet2tracksCALO;
     jet2tracksCALO.getByLabel(ev,"ic5JetToTracksAssociatorAtCaloFace");
     fwlite::Handle<std::vector <std::pair <edm::RefToBase<reco::Jet>, reco::JetExtendedAssociation::JetExtendedData> > > jetExtend;
     jetExtend.getByLabel(ev,"ic5JetExtender");
     //now can access data
     std::cout << "Total jets: " << jets->size() << std::endl;
     for (unsigned j = 0; j < jets->size(); ++j) {
       std::cout << "Jet #" << j << std::endl
		 << ((*jets)[j]).print() << std::endl;

       std::cout << "Associated tracks: " << reco::JetToTracksAssociation::tracksNumber (*jet2tracksVX, (*jets)[j]) 
		 << ", sumPt: " << reco::JetToTracksAssociation::tracksP4 (*jet2tracksVX, (*jets)[j]).pt() << std::endl;
       reco::TrackRefVector tracks = reco::JetToTracksAssociation::getValue (*jet2tracksVX, (*jets)[j]);
       for (unsigned t = 0; t < tracks.size(); ++t) {
	 std::cout << "track p/pt/eta/phi: " << (tracks[t])->p() << '/' << (tracks[t])->pt() << '/' 
		   << (tracks[t])->eta() << '/' << (tracks[t])->phi() << std::endl;
       }

       std::cout << "Associated tracks at CALO face: " << reco::JetToTracksAssociation::tracksNumber (*jet2tracksCALO, (*jets)[j])
		 << ", sumPt: " << reco::JetToTracksAssociation::tracksP4 (*jet2tracksCALO, (*jets)[j]).pt() << std::endl;
       reco::TrackRefVector tracksAtFace = reco::JetToTracksAssociation::getValue (*jet2tracksCALO, (*jets)[j]);
       for (unsigned t = 0; t < tracksAtFace.size(); ++t) {
	 std::cout << "track p/pt/eta/phi: " << (tracksAtFace[t])->p() << '/' << (tracksAtFace[t])->pt() << '/' 
		   << (tracksAtFace[t])->eta() << '/' << (tracksAtFace[t])->phi() << std::endl;
       }

       std::cout << "Jet extended information:"
		 << " in VX tracks " << reco::JetExtendedAssociation::tracksInVertexNumber (*jetExtend, (*jets)[j])
		 << ", sumPt: " << reco::JetExtendedAssociation::tracksInVertexP4 (*jetExtend, (*jets)[j]).pt()
		 << "; at CALO tracks " << reco::JetExtendedAssociation::tracksAtCaloNumber (*jetExtend, (*jets)[j])
		 << ", sumPt: " << reco::JetExtendedAssociation::tracksAtCaloP4 (*jetExtend, (*jets)[j]).pt()
		 << std::endl;
       
     }
     std::cout << "=================== event end ====================" << std::endl;
   }
}
