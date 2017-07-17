#ifndef DTTimeEvolutionHisto_H
#define DTTimeEvolutionHisto_H

/** \class DTTimeEvolutionHisto
 *  No description available.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include <string>
#include <map>

#include "DQMServices/Core/interface/DQMStore.h"

class DQMStore;
class MonitorElement;

class DTTimeEvolutionHisto {
public:
  /// Constructor
  /// Parameters are: <br>
  ///    - pointer to DQMStore <br>
  ///    - name of the MonitorElement <br>
  ///    - title of the MonitorElement <br>
  ///    - # of bins <br>
  ///    - # of LumiSections per bin <br>
  ///    - mode: <br>
  ///         0 -> rate (over event) <br>
  ///              need to fill using accumulateValueTimeSlot and updateTimeSlot methods <br>
  ///         1 -> # of entries <br>
  ///         2 -> # of events <br>
  ///         3 -> mean over LSs <br>
  DTTimeEvolutionHisto(DQMStore::IBooker& ibooker, const std::string& name,
		       const std::string& title,
		       int nbins,
		       int lsPrescale,
		       bool sliding,
		       int mode = 0);


  DTTimeEvolutionHisto(DQMStore::IBooker& ibooker, const std::string& name,
		       const std::string& title,
		       int nbins,
		       int firstLS,
		       int lsPrescale,
		       bool sliding,
		       int mode = 0);


  //FR changed the previous 2 argument constructor to the following one
  DTTimeEvolutionHisto(MonitorElement*);

  /// Destructor
  virtual ~DTTimeEvolutionHisto();

  // Operations

  void setTimeSlotValue(float value, int timeSlot);

  void accumulateValueTimeSlot(float value);

  void updateTimeSlot(int ls, int nEventsInLS);

  void normalizeTo(const MonitorElement *histForNorm);

protected:

private:
  float valueLastTimeSlot;
  std::map<int,int> nEventsInLastTimeSlot;
  std::map<int,int> nLumiTrInLastTimeSlot;
  int theFirstLS;
  int theLSPrescale;
  bool doSlide;
  int nBookedBins;
  int theMode;
  MonitorElement *histo;

};
#endif


/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
