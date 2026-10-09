#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"

const std::unordered_map<HGCalTriggerGeoTesterErrors::ErrorCode, std::string> HGCalTriggerGeoTesterErrors::messages = {
    {HGCalTriggerGeoTesterErrors::CellValidity, "Found invalid cell(s)"},
    {HGCalTriggerGeoTesterErrors::MissingCellInTC, "Found missing cell(s) in trigger cell"},
    {HGCalTriggerGeoTesterErrors::InvalidCellInTC, "Found invalid cell(s) in trigger cell"},
    {HGCalTriggerGeoTesterErrors::MissingTCInModule, "Found missing trigger cell(s) in module"},
    {HGCalTriggerGeoTesterErrors::InvalidTCInModule, "Found invalid trigger cell(s) in module"},
    {HGCalTriggerGeoTesterErrors::ConnectedModuleWithoutLpgbt, "Found module without lpGBT but flagged as connected"},
    {HGCalTriggerGeoTesterErrors::ModuleSplitInStage1, "Found module with lpGBTs connected to several Stage 1 FPGAs"},
    {HGCalTriggerGeoTesterErrors::MissingModuleInStage1FPGA, "Found missing module(s) in Stage 1 FPGA"},
    {HGCalTriggerGeoTesterErrors::InvalidModuleInStage1FPGA, "Found invalid module(s) in Stage 1 FPGA"},
    {HGCalTriggerGeoTesterErrors::MissingStage1InStage2FPGA, "Found missing stage1(s) in Stage 2 FPGA"},
    {HGCalTriggerGeoTesterErrors::InvalidStage1InStage2FPGA, "Found invalid stage1(s) in Stage 2 FPGA"}};

EDM_REGISTER_PLUGINFACTORY(HGCalTriggerGeoTesterFactory, "HGCalTriggerGeoTesterFactory");
