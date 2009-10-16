#ifndef MuonIdentification_TimeMeasurementSequence_h
#define MuonIdentification_TimeMeasurementSequence_h

/** \class reco::TimeMeasurementSequence TimeMeasurementSequence.h RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h
 *  
 * A class holding a set of individual time measurements along the muon trajectory
 *
 * \author Piotr Traczyk, CERN
 *
 * \version $Id: TimeMeasurementSequence.h,v 1.2 2009/03/27 02:26:41 ptraczyk Exp $
 *
 */

class TimeMeasurementSequence {

    public:

      std::vector <double> dstnc;
      std::vector <double> local_t0;
      std::vector <double> weight;
      
      double totalWeight;
      
      TimeMeasurementSequence():
       totalWeight(0)
	 {}

};


#endif
