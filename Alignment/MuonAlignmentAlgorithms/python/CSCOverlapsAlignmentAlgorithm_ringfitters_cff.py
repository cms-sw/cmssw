import FWCore.ParameterSet.Config as cms

# fitter configuration (no constraints)
fitters = cms.VPSet(

cms.PSet(name = cms.string("ME+1/1"),
         alignables = cms.vstring("ME+1/1/01", "ME+1/1/02", "ME+1/1/03", "ME+1/1/04", "ME+1/1/05", "ME+1/1/06", "ME+1/1/07", "ME+1/1/08", "ME+1/1/09", "ME+1/1/10", "ME+1/1/11", "ME+1/1/12", "ME+1/1/13", "ME+1/1/14", "ME+1/1/15", "ME+1/1/16", "ME+1/1/17", "ME+1/1/18", "ME+1/1/19", "ME+1/1/20", "ME+1/1/21", "ME+1/1/22", "ME+1/1/23", "ME+1/1/24", "ME+1/1/25", "ME+1/1/26", "ME+1/1/27", "ME+1/1/28", "ME+1/1/29", "ME+1/1/30", "ME+1/1/31", "ME+1/1/32", "ME+1/1/33", "ME+1/1/34", "ME+1/1/35", "ME+1/1/36"),
         fixed = cms.string(""),
         constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+1/2"),
             alignables = cms.vstring("ME+1/2/01", "ME+1/2/02", "ME+1/2/03", "ME+1/2/04", "ME+1/2/05", "ME+1/2/06", "ME+1/2/07", "ME+1/2/08", "ME+1/2/09", "ME+1/2/10", "ME+1/2/11", "ME+1/2/12", "ME+1/2/13", "ME+1/2/14", "ME+1/2/15", "ME+1/2/16", "ME+1/2/17", "ME+1/2/18", "ME+1/2/19", "ME+1/2/20", "ME+1/2/21", "ME+1/2/22", "ME+1/2/23", "ME+1/2/24", "ME+1/2/25", "ME+1/2/26", "ME+1/2/27", "ME+1/2/28", "ME+1/2/29", "ME+1/2/30", "ME+1/2/31", "ME+1/2/32", "ME+1/2/33", "ME+1/2/34", "ME+1/2/35", "ME+1/2/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+2/1"),
             alignables = cms.vstring("ME+2/1/01", "ME+2/1/02", "ME+2/1/03", "ME+2/1/04", "ME+2/1/05", "ME+2/1/06", "ME+2/1/07", "ME+2/1/08", "ME+2/1/09", "ME+2/1/10", "ME+2/1/11", "ME+2/1/12", "ME+2/1/13", "ME+2/1/14", "ME+2/1/15", "ME+2/1/16", "ME+2/1/17", "ME+2/1/18"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+2/2"),
             alignables = cms.vstring("ME+2/2/01", "ME+2/2/02", "ME+2/2/03", "ME+2/2/04", "ME+2/2/05", "ME+2/2/06", "ME+2/2/07", "ME+2/2/08", "ME+2/2/09", "ME+2/2/10", "ME+2/2/11", "ME+2/2/12", "ME+2/2/13", "ME+2/2/14", "ME+2/2/15", "ME+2/2/16", "ME+2/2/17", "ME+2/2/18", "ME+2/2/19", "ME+2/2/20", "ME+2/2/21", "ME+2/2/22", "ME+2/2/23", "ME+2/2/24", "ME+2/2/25", "ME+2/2/26", "ME+2/2/27", "ME+2/2/28", "ME+2/2/29", "ME+2/2/30", "ME+2/2/31", "ME+2/2/32", "ME+2/2/33", "ME+2/2/34", "ME+2/2/35", "ME+2/2/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+3/1"),
             alignables = cms.vstring("ME+3/1/01", "ME+3/1/02", "ME+3/1/03", "ME+3/1/04", "ME+3/1/05", "ME+3/1/06", "ME+3/1/07", "ME+3/1/08", "ME+3/1/09", "ME+3/1/10", "ME+3/1/11", "ME+3/1/12", "ME+3/1/13", "ME+3/1/14", "ME+3/1/15", "ME+3/1/16", "ME+3/1/17", "ME+3/1/18"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+3/2"),
             alignables = cms.vstring("ME+3/2/01", "ME+3/2/02", "ME+3/2/03", "ME+3/2/04", "ME+3/2/05", "ME+3/2/06", "ME+3/2/07", "ME+3/2/08", "ME+3/2/09", "ME+3/2/10", "ME+3/2/11", "ME+3/2/12", "ME+3/2/13", "ME+3/2/14", "ME+3/2/15", "ME+3/2/16", "ME+3/2/17", "ME+3/2/18", "ME+3/2/19", "ME+3/2/20", "ME+3/2/21", "ME+3/2/22", "ME+3/2/23", "ME+3/2/24", "ME+3/2/25", "ME+3/2/26", "ME+3/2/27", "ME+3/2/28", "ME+3/2/29", "ME+3/2/30", "ME+3/2/31", "ME+3/2/32", "ME+3/2/33", "ME+3/2/34", "ME+3/2/35", "ME+3/2/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+4/1"),
             alignables = cms.vstring("ME+4/1/01", "ME+4/1/02", "ME+4/1/03", "ME+4/1/04", "ME+4/1/05", "ME+4/1/06", "ME+4/1/07", "ME+4/1/08", "ME+4/1/09", "ME+4/1/10", "ME+4/1/11", "ME+4/1/12", "ME+4/1/13", "ME+4/1/14", "ME+4/1/15", "ME+4/1/16", "ME+4/1/17", "ME+4/1/18"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME+4/2"),
             alignables = cms.vstring("ME+4/2/09", "ME+4/2/10", "ME+4/2/11", "ME+4/2/12", "ME+4/2/13"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-1/1"),
             alignables = cms.vstring("ME-1/1/01", "ME-1/1/02", "ME-1/1/03", "ME-1/1/04", "ME-1/1/05", "ME-1/1/06", "ME-1/1/07", "ME-1/1/08", "ME-1/1/09", "ME-1/1/10", "ME-1/1/11", "ME-1/1/12", "ME-1/1/13", "ME-1/1/14", "ME-1/1/15", "ME-1/1/16", "ME-1/1/17", "ME-1/1/18", "ME-1/1/19", "ME-1/1/20", "ME-1/1/21", "ME-1/1/22", "ME-1/1/23", "ME-1/1/24", "ME-1/1/25", "ME-1/1/26", "ME-1/1/27", "ME-1/1/28", "ME-1/1/29", "ME-1/1/30", "ME-1/1/31", "ME-1/1/32", "ME-1/1/33", "ME-1/1/34", "ME-1/1/35", "ME-1/1/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-1/2"),
             alignables = cms.vstring("ME-1/2/01", "ME-1/2/02", "ME-1/2/03", "ME-1/2/04", "ME-1/2/05", "ME-1/2/06", "ME-1/2/07", "ME-1/2/08", "ME-1/2/09", "ME-1/2/10", "ME-1/2/11", "ME-1/2/12", "ME-1/2/13", "ME-1/2/14", "ME-1/2/15", "ME-1/2/16", "ME-1/2/17", "ME-1/2/18", "ME-1/2/19", "ME-1/2/20", "ME-1/2/21", "ME-1/2/22", "ME-1/2/23", "ME-1/2/24", "ME-1/2/25", "ME-1/2/26", "ME-1/2/27", "ME-1/2/28", "ME-1/2/29", "ME-1/2/30", "ME-1/2/31", "ME-1/2/32", "ME-1/2/33", "ME-1/2/34", "ME-1/2/35", "ME-1/2/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-2/1"),
             alignables = cms.vstring("ME-2/1/01", "ME-2/1/02", "ME-2/1/03", "ME-2/1/04", "ME-2/1/05", "ME-2/1/06", "ME-2/1/07", "ME-2/1/08", "ME-2/1/09", "ME-2/1/10", "ME-2/1/11", "ME-2/1/12", "ME-2/1/13", "ME-2/1/14", "ME-2/1/15", "ME-2/1/16", "ME-2/1/17", "ME-2/1/18"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-2/2"),
             alignables = cms.vstring("ME-2/2/01", "ME-2/2/02", "ME-2/2/03", "ME-2/2/04", "ME-2/2/05", "ME-2/2/06", "ME-2/2/07", "ME-2/2/08", "ME-2/2/09", "ME-2/2/10", "ME-2/2/11", "ME-2/2/12", "ME-2/2/13", "ME-2/2/14", "ME-2/2/15", "ME-2/2/16", "ME-2/2/17", "ME-2/2/18", "ME-2/2/19", "ME-2/2/20", "ME-2/2/21", "ME-2/2/22", "ME-2/2/23", "ME-2/2/24", "ME-2/2/25", "ME-2/2/26", "ME-2/2/27", "ME-2/2/28", "ME-2/2/29", "ME-2/2/30", "ME-2/2/31", "ME-2/2/32", "ME-2/2/33", "ME-2/2/34", "ME-2/2/35", "ME-2/2/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-3/1"),
             alignables = cms.vstring("ME-3/1/01", "ME-3/1/02", "ME-3/1/03", "ME-3/1/04", "ME-3/1/05", "ME-3/1/06", "ME-3/1/07", "ME-3/1/08", "ME-3/1/09", "ME-3/1/10", "ME-3/1/11", "ME-3/1/12", "ME-3/1/13", "ME-3/1/14", "ME-3/1/15", "ME-3/1/16", "ME-3/1/17", "ME-3/1/18"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-3/2"),
             alignables = cms.vstring("ME-3/2/01", "ME-3/2/02", "ME-3/2/03", "ME-3/2/04", "ME-3/2/05", "ME-3/2/06", "ME-3/2/07", "ME-3/2/08", "ME-3/2/09", "ME-3/2/10", "ME-3/2/11", "ME-3/2/12", "ME-3/2/13", "ME-3/2/14", "ME-3/2/15", "ME-3/2/16", "ME-3/2/17", "ME-3/2/18", "ME-3/2/19", "ME-3/2/20", "ME-3/2/21", "ME-3/2/22", "ME-3/2/23", "ME-3/2/24", "ME-3/2/25", "ME-3/2/26", "ME-3/2/27", "ME-3/2/28", "ME-3/2/29", "ME-3/2/30", "ME-3/2/31", "ME-3/2/32", "ME-3/2/33", "ME-3/2/34", "ME-3/2/35", "ME-3/2/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    cms.PSet(name = cms.string("ME-4/1"),
             alignables = cms.vstring("ME-4/1/01", "ME-4/1/02", "ME-4/1/03", "ME-4/1/04", "ME-4/1/05", "ME-4/1/06", "ME-4/1/07", "ME-4/1/08", "ME-4/1/09", "ME-4/1/10", "ME-4/1/11", "ME-4/1/12", "ME-4/1/13", "ME-4/1/14", "ME-4/1/15", "ME-4/1/16", "ME-4/1/17", "ME-4/1/18"),
             fixed = cms.string(""),
             constraints = cms.VPSet()),

    )
