// Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
#ifndef CMSSW_PERFETTO_CATEGORIES_H
#define CMSSW_PERFETTO_CATEGORIES_H
#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(perfetto::Category("cmssw.event"),
                           perfetto::Category("cmssw.source"),
                           perfetto::Category("cmssw.module"),
                           perfetto::Category("cmssw.acquire"),
                           perfetto::Category("cmssw.cleanup"),
                           perfetto::Category("cmssw.es"),
                           perfetto::Category("cmssw.func"),
                           perfetto::Category("cmssw.alloc"),
                           perfetto::Category("cmssw.gpu"),
                           perfetto::Category("cmssw.power"));

#endif
