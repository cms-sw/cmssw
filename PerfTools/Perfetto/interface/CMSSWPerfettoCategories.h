// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef PerfTools_Perfetto_interface_CMSSWPerfettoCategories_h
#define PerfTools_Perfetto_interface_CMSSWPerfettoCategories_h

#include <perfetto.h>

// The cmssw.* track-event categories. They live in cms::perfetto rather than in
// perfetto's single global set, so other perfetto users in the process cannot
// clash with them; the TRACE_* macros pick them up through the using-declaration.
PERFETTO_DEFINE_CATEGORIES_IN_NAMESPACE(cms::perfetto,
                                        ::perfetto::Category("cmssw.event"),
                                        ::perfetto::Category("cmssw.source"),
                                        ::perfetto::Category("cmssw.module"),
                                        ::perfetto::Category("cmssw.acquire"),
                                        ::perfetto::Category("cmssw.cleanup"),
                                        ::perfetto::Category("cmssw.es"),
                                        ::perfetto::Category("cmssw.func"),
                                        ::perfetto::Category("cmssw.alloc"),
                                        ::perfetto::Category("cmssw.gpu"),
                                        ::perfetto::Category("cmssw.power"));
PERFETTO_USE_CATEGORIES_FROM_NAMESPACE(cms::perfetto);

#endif  // PerfTools_Perfetto_interface_CMSSWPerfettoCategories_h
