import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

primaryVertexResolutionClient = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("OfflinePV/Resolution/*"),
    efficiency = cms.vstring(),
    resolution = cms.vstring(
        "res_x_vs_ntracks '#sigma(x) vs ntracks' res_x_vs_ntracks",
        "res_y_vs_ntracks '#sigma(y) vs ntracks' res_y_vs_ntracks",
        "res_z_vs_ntracks '#sigma(z) vs ntracks' res_z_vs_ntracks",
        "pull_x_vs_ntracks 'x pull vs ntracks' pull_x_vs_ntracks",
        "pull_y_vs_ntracks 'y pull vs ntracks' pull_y_vs_ntracks",
        "pull_z_vs_ntracks 'z pull vs ntracks' pull_z_vs_ntracks",

        "res_x_vs_sumpt '#sigma(x) vs sumpt' res_x_vs_sumpt",
        "res_y_vs_sumpt '#sigma(y) vs sumpt' res_y_vs_sumpt",
        "res_z_vs_sumpt '#sigma(z) vs sumpt' res_z_vs_sumpt",
        "pull_x_vs_sumpt 'x pull vs sumpt' pull_x_vs_sumpt",
        "pull_y_vs_sumpt 'y pull vs sumpt' pull_y_vs_sumpt",
        "pull_z_vs_sumpt 'z pull vs sumpt' pull_z_vs_sumpt",

        "res_x_vs_nvertices '#sigma(x) vs nvertices' res_x_vs_nvertices",
        "res_y_vs_nvertices '#sigma(y) vs nvertices' res_y_vs_nvertices",
        "res_z_vs_nvertices '#sigma(z) vs nvertices' res_z_vs_nvertices",
        "pull_x_vs_nvertices 'x pull vs nvertices' pull_x_vs_nvertices",
        "pull_y_vs_nvertices 'y pull vs nvertices' pull_y_vs_nvertices",
        "pull_z_vs_nvertices 'z pull vs nvertices' pull_z_vs_nvertices",

        "res_x_vs_instLumiScal '#sigma(x) vs instLumiScal' res_x_vs_instLumiScal",
        "res_y_vs_instLumiScal '#sigma(y) vs instLumiScal' res_y_vs_instLumiScal",
        "res_z_vs_instLumiScal '#sigma(z) vs instLumiScal' res_z_vs_instLumiScal",
        "pull_x_vs_instLumiScal 'x pull vs instLumiScal' pull_x_vs_instLumiScal",
        "pull_y_vs_instLumiScal 'y pull vs instLumiScal' pull_y_vs_instLumiScal",
        "pull_z_vs_instLumiScal 'z pull vs instLumiScal' pull_z_vs_instLumiScal",

    )
)

