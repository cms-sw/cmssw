#ifndef MuonReco_TimeMeasurementSequence_h
#define MuonReco_TimeMeasurementSequence_h

/** \class reco::TimeMeasurementSequence TimeMeasurementSequence.h RecoMuon/MuonTiming/interface/TimeMeasurementSequence.h
 *  
 * A class holding a set of individual time measurements along the muon trajectory
 *
 * \author Piotr Traczyk, CERN
 *
 * \version $Id: TimeMeasurementSequence.h,v 1.1 2009/03/13 22:58:14 ptraczyk Exp $
 *
 */

class TimeMeasurementSequence {

    public:

      std::vector <double> dstnc;
      std::vector <double> local_t0;
      std::vector <double> weight;
      
      int totalWeight;
      
      TimeMeasurementSequence():
       totalWeight(0)
	 {}

};


#endif
