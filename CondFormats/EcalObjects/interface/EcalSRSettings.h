/*
 * $Id: EcalSRSettings.h,v 1.2 2010/06/14 10:45:15 pgras Exp $
 *
 * Original author: Ph. Gras CEA/IRFU Saclay.  June, 2010
 *
 */

#ifndef ECALSRSETTINGS_H
#define ECALSRSETTINGS_H

#include <vector>
#include <string>
#include <ostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** Class to hold ECAL Selective readout settings.
 * Up to CMSSW release 3.8.X, selective readout emulation settings was done from
 * CMSSW configuration file. From 3.8.X configuration is stored in condition database.
 * Support for configuration from CMSSW configuration file is maintained, for backward
 * compatibility and to provide an easy way to change settings for studies dedicated to
 * seletive readout. The same object is used to stored the online setting of real data.
 *
 * Tools to manipulate this object can be found in class EcalSRCondTools of package
 * SimCalorimetry/EcalSelectiveReadoutProducer.
 */
class EcalSRSettings {

  //constructor(s) and destructor(s)
public:
  /** Constructs an EcalSRSettings
   */
  EcalSRSettings();

  /**Destructor
   */
  virtual ~EcalSRSettings(){};

  //method(s)
public:

private:
  //attribute(s)
protected:
private:
public:
  static const int nSrps_ = 12;
  static const int nDccs_ = 54;
  static const int nTccs_ = 108;
  
  /// Neighbour eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
  /// In the vector contains:
  ///   - 1 element, then value applies to whole ECAL
  ///   - 2 elements, then element 0 applies to EB, element 1 to EE
  ///   - 12 elements, then element i applied to SRP (i+1)
  /// SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports
  /// only 1 element mode.
  std::vector<int> deltaEta_;
    
  /// Neighbouring eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
  /// If the vector contains...
  ///   ... 1 element, then value applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 12 elements, then element i applied to SRP (i+1)
  /// If the vector contains...
  ///   ... 1 element, then value applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 12 elements, then element i applied to SRP (i+1)
  /// SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports
  /// only the single-element mode.
  std::vector<int> deltaPhi_;
    
  /// Index of time sample (staring from 1) the first DCC weights is implied
  /// If the vector contains:
  ///   ... 1 element, then value applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)
  /// SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports
  /// only the single-element mode.
  std::vector<int> ecalDccZs1stSample_;

  /// ADC to GeV conversion factor used in ZS filter for EB
  float ebDccAdcToGeV_;
    /// ADC to GeV conversion factor used in ZS filter for EE
  float eeDccAdcToGeV_;

  ///DCC ZS FIR weights: weights are rounded in such way that in Hw
  ///representation (weigth*1024 rounded to nearest integer) the sum is null:
  ///Each element is a vector of 6 values, the 6 weights
  /// If the vector contains...
  ///   ... 1 element, then the weight set applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)
  ///   ... 75848 elements, then:
  ///            for i < 61200, element i applies to EB crystal with denseIndex i
  ///                           (see EBDetId::denseIndex())
  ///            for i >= 61200, element i applies to EE crystal with denseIndex (i+61200)
  ///                           (see EBDetId::denseIndex())
  std::vector<std::vector<float> > dccNormalizedWeights_;
  
  /// Switch to use a symetric zero suppression (cut on absolute value). For
  /// studies only, for time being it is not supported by the hardware.
  /// having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead,
  /// 0 means false, a value different than 0 means true.
  /// If the vector contains...
  ///   ... 1 element, then the weight set applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)
  /// SRP emulation supports only 1 element mode. Hardware does not support
  /// the symetric ZS, so symetricZS = 0 for real data.
  std::vector<int> symetricZS_;

  /// ZS energy threshold in GeV to apply to low interest channels of barrel
  /// If the vector contains...
  ///   ... 1 element, then the weight set applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)
  /// SRP emulation supports only the 2-element mode.
  /// Corresponds to srpBarrelLowInterestChannelZS and srpEndcapLowInterestChannelZS
  /// of python configuration file parameters
  std::vector<float> srpLowInterestChannelZS_;
    
  /// ZS energy threshold in GeV to apply to high interest channels of endcap
  /// If the vector contains...
  ///   ... 1 element, then the weight set applies to whole ECAL
  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
  ///   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)
  /// SRP emulation supports only the 2-element mode.
  /// Corresponds to srpBarrelLowInterestChannelZS and srpEndcapLowInterestChannelZS
  /// of python configuration file parameters
  std::vector<float> srpHighInterestChannelZS_;
  
//  ///switch to run w/o trigger primitive. For debug use only
//  ///having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead
//  ///Parameter only relevant for emulation. For real data, must be contains 1 element with
//  ///value 0.
//  ///   ... 1 element, then the weight set applies to whole ECAL
//  ///   ... 2 elements, then element 0 applies to EB, element 1 to EE
//  ///   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)
//  /// SRP emulation supports only the single-element mode.  
//  std::vector<int> trigPrimBypass_;
//  
//  /// Mode selection for "Trig bypass" mode
//  /// 0: TT thresholds applied on sum of crystal Et's
//  /// 1: TT thresholds applies on compressed Et from Trigger primitive
//  /// @see trigPrimByPass switch
//  /// Parameter only relevant for 
//  std::vector<int> trigPrimBypassMode_;
//  
//  ///for debug mode only:
//  std::vector<float>  trigPrimBypassLTH_;
//  
//  ///for debug mode only:
//  std::vector<float>  trigPrimBypassHTH_;
//    
//  ///for debug mode only
//  ///having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead
//  std::vector<int>  trigPrimBypassWithPeakFinder_;
//  
//  ///Trigger Tower Flag to use when a flag is not found from the input
//  ///Trigger Primitive collection. Must be one of the following values:
//  /// 0: low interest, 1: mid interest, 3: high interest
//  /// 4: forced low interest, 5: forced mid interest, 7: forced high interest
//  std::vector<int> defaultTtf_;
  
  /// SR->action flag map. 4 elements
  /// action_[i]: action for flag value i
  std::vector<int> actions_;

  ///Masks for TTC inputs of SRP cards
  ///One element per TCC, that is 108 elements: element i applies to TCC (i+1)
  std::vector<short> tccMasksFromConfig_;

  ///Masks for SRP-SRP inputs of SRP cards
  ///One element per SRP, that is 12 elements: element i applies to SRP (i+1)
  // indices: [iSrp][iCh]
  std::vector<std::vector<short> >srpMasksFromConfig_;

  ///Masks for DCC output of SRP cards
  ///One element per DCC, that is 54 elements: element i applies to DCC (i+1)
  std::vector<short> dccMasks_;

  ///Mask to enable pattern test. Typical value: 0.
  ///One element per SRP, that is 12 elements: element i applies to SRP (i+1)
  std::vector<short> srfMasks_;

  ///Substitution flags used in patterm mode
  ///indices [iSrp][iFlag]
  std::vector<std::vector<short> >substitutionSrfs_;

  //@{
  ///Tester mode configuration
  /// 12 elements: index = (SRP ID - 1)
  std::vector<int> testerTccEmuSrpIds_;
  std::vector<int> testerSrpEmuSrpIds_;
  std::vector<int> testerDccTestSrpIds_;
  std::vector<int> testerSrpTestSrpIds_;
  //@}
  
  ///Per SRP card bunch crossing counter offset.
  ///This offset is added to the bxGlobalOffset
  /// 12 elements: index = (SRP ID - 1)
  std::vector<short> bxOffsets_;

  ///SRP system bunch crossing counter offset.
  ///For each card the bxOffset[i] (typ. value 0)
  //is added to this one.
  short bxGlobalOffset_;

  /// Switch for automatic channel masking. 0: disabled; 1: enabled. Standard  configuration: 1.
  /// When enabled, if a FED is excluded from the run, the corresponding TCC inputs is automatically
  /// masked (overwrites the tccInputMasks).
  int automaticMasks_;

  /// Switch for automatic SRP card selection. 0: disabled; 1 : enabled..
  ///When enabled, if all the FEDs corresponding to a given SRP is excluded from the run,
  ///Then the corresponding SRP card is automatically excluded.
  int automaticSrpSelect_;
};

std::ostream& operator<< (std::ostream& o, const EcalSRSettings& val);

#endif //ECALSRSETTINGS_H not defined
