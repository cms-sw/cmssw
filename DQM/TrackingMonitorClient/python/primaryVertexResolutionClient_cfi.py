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

        "res_x_vs_X '#sigma(x) vs X' res_x_vs_X",
        "res_y_vs_X '#sigma(y) vs X' res_y_vs_X",
        "res_z_vs_X '#sigma(z) vs X' res_z_vs_X",
        "pull_x_vs_X 'x pull vs X' pull_x_vs_X",
        "pull_y_vs_X 'y pull vs X' pull_y_vs_X",
        "pull_z_vs_X 'z pull vs X' pull_z_vs_X",

        "res_x_vs_Y '#sigma(x) vs Y' res_x_vs_Y",
        "res_y_vs_Y '#sigma(y) vs Y' res_y_vs_Y",
        "res_z_vs_Y '#sigma(z) vs Y' res_z_vs_Y",
        "pull_x_vs_Y 'x pull vs Y' pull_x_vs_Y",
        "pull_y_vs_Y 'y pull vs Y' pull_y_vs_Y",
        "pull_z_vs_Y 'z pull vs Y' pull_z_vs_Y",

        "res_x_vs_Z '#sigma(x) vs Z' res_x_vs_Z",
        "res_y_vs_Z '#sigma(y) vs Z' res_y_vs_Z",
        "res_z_vs_Z '#sigma(z) vs Z' res_z_vs_Z",
        "pull_x_vs_Z 'x pull vs Z' pull_x_vs_Z",
        "pull_y_vs_Z 'y pull vs Z' pull_y_vs_Z",
        "pull_z_vs_Z 'z pull vs Z' pull_z_vs_Z"
    )
)

