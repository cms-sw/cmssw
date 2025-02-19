#ifndef RecoJets_JetProducers_interface_VirtualJetProducerHelper_h
#define RecoJets_JetProducers_interface_VirtualJetProducerHelper_h



namespace reco {

  namespace helper {

    namespace VirtualJetProducerHelper {
      
      // Area of intersection of two unit-radius disks with centers separated by r12.
      double intersection(double r12) ;
  // Area of intersection of three unit-radius disks with centers separated by r12, r23, r13.
      double intersection(double r12, double r23, double r13) ;
    }
  }
}
#endif
