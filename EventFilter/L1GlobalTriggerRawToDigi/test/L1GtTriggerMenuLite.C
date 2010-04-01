/*
 * L1GtTriggerMenuLite.C
 *
 *  Created on: Feb 14, 2010
 *      Author: ghete
 */

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Run.h"
#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"

#if !defined(__CINT__) && !defined(__MAKECINT__)

// headers for the data items
#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

#endif

void L1GtTriggerMenuLite() {

    TFile file("L1GtTriggerMenuLite_output.root");

    fwlite::Run run(&file);

    for (run.toBegin(); !run.atEnd(); ++run) {

        //fwlite::Handle<L1GtTriggerMenuLite> trigMenu;
        //trigMenu.getbyLabel(ev, "l1GtTriggerMenuLite");
    }

    std::cout << "\nEnd FWLite\n";

}
