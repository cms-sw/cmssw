{
   #include "DataFormats/FWLite/interface/Handle.h"

   TFile file("/scratch/ratnikov/data/RelVal152BJets50-120_17X_jettrackassociation2.root");

   fwlite::Event ev(&file);
   for( ev.toBegin();
          ! ev.atEnd();
          ++ev) {
     fwlite::Handle<reco::CaloJetCollection> jets;
     jets.getByLabel(ev,"iterativeCone5CaloJets");
     const reco::JetTracksAssociation::Container* jet2tracksVX = reco::JetTracksAssociation::getByLabel (ev,"ic5JetTracksAssociatorAtVertex", 0, 0);;
     const reco::JetTracksAssociation::Container* jet2tracksCALO = reco::JetTracksAssociation::getByLabel (ev,"ic5JetTracksAssociatorAtCaloFace", 0, 0);
     const reco::JetExtendedAssociation::Container* jetExtend = reco::JetExtendedAssociation::getByLabel (ev,"ic5JetExtender", 0, 0);
   //now can access data
   std::cout << "Total jets: " << jets->size() << std::endl;
     for (unsigned j = 0; j < jets->size(); ++j) {
       std::cout << "Jet #" << j << std::endl
		 << ((*jets)[j]).print() << std::endl;
       
       std::cout << "Associated tracks at VX: " << reco::JetTracksAssociation::tracksNumber (*jet2tracksVX, (*jets)[j]) 
		 << ", sumPt: " << reco::JetTracksAssociation::tracksP4 (*jet2tracksVX, (*jets)[j]).pt() << std::endl;
       reco::TrackRefVector tracks = reco::JetTracksAssociation::getValue (*jet2tracksVX, (*jets)[j]);
       for (unsigned t = 0; t < tracks.size(); ++t) {
	 std::cout << "track p/pt/eta/phi: " << (tracks[t])->p() << '/' << (tracks[t])->pt() << '/' 
		   << (tracks[t])->eta() << '/' << (tracks[t])->phi() << std::endl;
       }
       
       std::cout << "Associated tracks at CALO face: " << reco::JetTracksAssociation::tracksNumber (*jet2tracksCALO, (*jets)[j])
		 << ", sumPt: " << reco::JetTracksAssociation::tracksP4 (*jet2tracksCALO, (*jets)[j]).pt() << std::endl;
       reco::TrackRefVector tracksAtFace = reco::JetTracksAssociation::getValue (*jet2tracksCALO, (*jets)[j]);
       for (unsigned t = 0; t < tracksAtFace.size(); ++t) {
	 std::cout << "track p/pt/eta/phi: " << (tracksAtFace[t])->p() << '/' << (tracksAtFace[t])->pt() << '/' 
		   << (tracksAtFace[t])->eta() << '/' << (tracksAtFace[t])->phi() << std::endl;
       }
       
       std::cout << "Jet extended information: " 
		 << " at VX tracks " << reco::JetExtendedAssociation::tracksAtVertexNumber (*jetExtend, (*jets)[j])
		 << ", sumPt: " << reco::JetExtendedAssociation::tracksAtVertexP4 (*jetExtend, (*jets)[j]).pt()
		 << "; at CALO tracks " << reco::JetExtendedAssociation::tracksAtCaloNumber (*jetExtend, (*jets)[j])
		 << ", sumPt: " << reco::JetExtendedAssociation::tracksAtCaloP4 (*jetExtend, (*jets)[j]).pt()
		 << std::endl;
     }
     std::cout << "=================== event end ====================" << std::endl;
   }
}
