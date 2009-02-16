// $Id: $
#ifndef HISTOOUTPUT_H 
#define HISTOOUTPUT_H 1

// Include files
#include <string>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

//From ROOT:
#include <TH1D.h>
#include <TH2D.h>

/** @class HistoOutput HistoOutput.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-02-05
 */
class HistoOutput {
public: 
  /// Standard constructor
  HistoOutput( ) {};

  HistoOutput( edm::Service<TFileService> &, const char * );

  virtual ~HistoOutput( ); ///< Destructor
  
  TH1D * hcompflag;
  TH1D * hnumGMTmuons;
  TH1D * htrigdecision;
  TH2D * hhtrgperevent;
    
protected:
  
private:
  
  std::string m_option;
  
};
#endif // HISTOOUTPUT_H
