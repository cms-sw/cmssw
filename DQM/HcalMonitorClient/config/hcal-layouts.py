def hcallayout(i, p, *rows): i["Layouts/HCAL Layouts/" + p] = DQMItem(layout=rows)

hcallayout(dqmitems, "HCAL Data Format Summary",
           ["Hcal/DataFormatMonitor/HTR Error Word by Crate"],
           ["Hcal/DataFormatMonitor/Bad Quality Digis",
            "Hcal/DataFormatMonitor/Unmapped Digis"],
           ["Hcal/DataFormatMonitor/Event Fragments violating the Common Data Format",
            "Hcal/DataFormatMonitor/DCC Event Format violation",
            "Hcal/DataFormatMonitor/DCC Error and Warning",
            "Hcal/DataFormatMonitor/DCC View of Spigot Conditions"]
           )

hcallayout(dqmitems, "HCAL Digitization Summary",
           ["Hcal/DigiMonitor/Digi Depth 1 Occupancy Map",
            "Hcal/DigiMonitor/Digi Depth 2 Occupancy Map",
            "Hcal/DigiMonitor/Digi Depth 3 Occupancy Map",
            "Hcal/DigiMonitor/Digi Depth 4 Occupancy Map"],
           ["Hcal/DigiMonitor/HB/HB # of Digis",
            "Hcal/DigiMonitor/HE/HE # of Digis",
            "Hcal/DigiMonitor/HF/HF # of Digis",
            "Hcal/DigiMonitor/HO/HO # of Digis"],
           ["Hcal/DigiMonitor/HB/HB QIE ADC Value",
            "Hcal/DigiMonitor/HE/HE QIE ADC Value",
            "Hcal/DigiMonitor/HF/HF QIE ADC Value",
            "Hcal/DigiMonitor/HO/HO QIE ADC Value"]
           )

hcallayout(dqmitems, "HCAL Reconstruction Summary",
           ["Hcal/RecHitMonitor/RecHit Total Energy"],
           ["Hcal/RecHitMonitor/HB/HB RecHit Energies",
            "Hcal/RecHitMonitor/HE/HE RecHit Energies",
            "Hcal/RecHitMonitor/HF/HF RecHit Energies",
            "Hcal/RecHitMonitor/HO/HO RecHit Energies"],
           ["Hcal/RecHitMonitor/HB/HB RecHit Times",
            "Hcal/RecHitMonitor/HE/HE RecHit Times",
            "Hcal/RecHitMonitor/HF/HF RecHit Times",
            "Hcal/RecHitMonitor/HO/HO RecHit Times"]
           )

hcallayout(dqmitems, "HCAL Reconstruction Threshold Summary",
           ["Hcal/RecHitMonitor/RecHit Total Energy - Threshold"],
           ["Hcal/RecHitMonitor/HB/HB RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HE/HE RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HF/HF RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HO/HO RecHit Total Energy - Threshold"],
           ["Hcal/RecHitMonitor/HB/HB RecHit Times - Threshold",
            "Hcal/RecHitMonitor/HE/HE RecHit Times - Threshold",
            "Hcal/RecHitMonitor/HF/HF RecHit Times - Threshold",
            "Hcal/RecHitMonitor/HO/HO RecHit Times - Threshold"]
           )

hcallayout(dqmitems, "HCAL Hot Cell Summary",
           ["Hcal/HotCellMonitor/HotCellEnergy",
            "Hcal/HotCellMonitor/HotCellTime"],
           ["Hcal/HotCellMonitor/HB/HBHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HE/HEHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HF/HFHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HO/HOHotCellOCCmap_Thresh0"]
           )

hcallayout(dqmitems, "HCAL Hot Cell NADA Summary",
           ["Hcal/HotCellMonitor/NADA_NumHotCells",
            "Hcal/HotCellMonitor/NADA_NumNegCells"],
           ["Hcal/HotCellMonitor/HB/NADA_HB_OCC_MAP",
            "Hcal/HotCellMonitor/HE/NADA_HE_OCC_MAP",
            "Hcal/HotCellMonitor/HF/NADA_HF_OCC_MAP",
            "Hcal/HotCellMonitor/HO/NADA_HO_OCC_MAP"]
           )

hcallayout(dqmitems, "HCAL Dead Cell Summary",
           ["Hcal/DeadCellMonitor/HB/HB_deadADCOccupancyMap",
            "Hcal/DeadCellMonitor/HE/HE_deadADCOccupancyMap"],
           ["Hcal/DeadCellMonitor/HF/HF_deadADCOccupancyMap",
            "Hcal/DeadCellMonitor/HO/HO_deadADCOccupancyMap"],
           ["Hcal/DeadCellMonitor/HB/HB_CoolCell_belowPed",
            "Hcal/DeadCellMonitor/HE/HE_CoolCell_belowPed"
            "Hcal/DeadCellMonitor/HF/HF_CoolCell_belowPed"
            "Hcal/DeadCellMonitor/HO/HO_CoolCell_belowPed"],
           ["Hcal/DeadCellMonitor/HB/HB_NADA_CoolCellMap"
            "Hcal/DeadCellMonitor/HE/HE_NADA_CoolCellMap"
            "Hcal/DeadCellMonitor/HF/HF_NADA_CoolCellMap"
            "Hcal/DeadCellMonitor/HO/HO_NADA_CoolCellMap"]
           )

hcallayout(dqmitems, "HCAL Pedestal Summary",
           ["Hcal/PedestalMonitor/HB/HB Normalized RMS Values",
            "Hcal/PedestalMonitor/HE/HE Normalized RMS Values",
            "Hcal/PedestalMonitor/HF/HF Normalized RMS Values",
            "Hcal/PedestalMonitor/HO/HO Normalized RMS Values"],           
            ["Hcal/PedestalMonitor/HB/HB Subtracted Mean Values",
             "Hcal/PedestalMonitor/HE/HE Subtracted Mean Values",
             "Hcal/PedestalMonitor/HF/HF Subtracted Mean Values",
             "Hcal/PedestalMonitor/HO/HO Subtracted Mean Values"],
            ["Hcal/PedestalMonitor/HB/HB CapID RMS Variance",
             "Hcal/PedestalMonitor/HE/HE CapID RMS Variance",
             "Hcal/PedestalMonitor/HF/HF CapID RMS Variance",
             "Hcal/PedestalMonitor/HO/HO CapID RMS Variance"],
            ["Hcal/PedestalMonitor/HB/HB CapID Mean Variance",
             "Hcal/PedestalMonitor/HE/HE CapID Mean Variance",
             "Hcal/PedestalMonitor/HF/HF CapID Mean Variance",
             "Hcal/PedestalMonitor/HO/HO CapID Mean Variance"]
           )

hcallayout(dqmitems, "HCAL LED Summary",
           ["Hcal/LEDMonitor/HB/HB Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HE/HE Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HF/HF Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HO/HO Ped Subtracted Pulse Shape"],
           ["Hcal/LEDMonitor/HB/HB Average Pulse Shape",
            "Hcal/LEDMonitor/HE/HE Average Pulse Shape",
            "Hcal/LEDMonitor/HF/HF Average Pulse Shape",
            "Hcal/LEDMonitor/HO/HO Average Pulse Shape"],
           ["Hcal/LEDMonitor/HB/HB Average Pulse Time",
            "Hcal/LEDMonitor/HE/HE Average Pulse Time",
            "Hcal/LEDMonitor/HF/HF Average Pulse Time",
            "Hcal/LEDMonitor/HO/HO Average Pulse Time"],
           ["Hcal/LEDMonitor/LED Mean Energy Depth 1",
            "Hcal/LEDMonitor/LED Mean Energy Depth 2",
            "Hcal/LEDMonitor/LED Mean Energy Depth 3",
            "Hcal/LEDMonitor/LED Mean Energy Depth 4"]
)

hcallayout(dqmitems, "HCAL Barrel Summary",
           ["Hcal/DigiMonitor/HB/HB # of Digis",
            "Hcal/DigiMonitor/HB/HB QIE ADC Value"],
           ["Hcal/RecHitMonitor/HB/HB RecHit Energies",
            "Hcal/RecHitMonitor/HB/HB RecHit Times",
            "Hcal/RecHitMonitor/HB/HB RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HB/HB RecHit Times - Threshold"],
           ["Hcal/HotCellMonitor/HB/HBHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HB/NADA_HB_OCC_MAP"],
           ["Hcal/DeadCellMonitor/HB/HB_deadADCOccupancyMap",
            "Hcal/DeadCellMonitor/HB/HB_CoolCell_belowPed",
            "Hcal/DeadCellMonitor/HB/HB_NADA_CoolCellMap"],
           ["Hcal/PedestalMonitor/HB/HB Normalized RMS Values",
            "Hcal/PedestalMonitor/HB/HB Subtracted Mean Values",
            "Hcal/PedestalMonitor/HB/HB CapID RMS Variance",
            "Hcal/PedestalMonitor/HB/HB CapID Mean Variance"],
           ["Hcal/LEDMonitor/HB/HB Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HB/HB Average Pulse Shape",
            "Hcal/LEDMonitor/HB/HB Average Pulse Time"]
)

hcallayout(dqmitems, "HCAL Endcap Summary",
           ["Hcal/DigiMonitor/HE/HE # of Digis",
            "Hcal/DigiMonitor/HE/HE QIE ADC Value"],
           ["Hcal/RecHitMonitor/HE/HE RecHit Energies",
            "Hcal/RecHitMonitor/HE/HE RecHit Times",
            "Hcal/RecHitMonitor/HE/HE RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HE/HE RecHit Times - Threshold"],
           ["Hcal/HotCellMonitor/HE/HEHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HE/NADA_HE_OCC_MAP"],
           ["Hcal/DeadCellMonitor/HE/HE_deadADCOccupancyMap",
            "Hcal/DeadCellMonitor/HE/HE_CoolCell_belowPed",
            "Hcal/DeadCellMonitor/HE/HE_NADA_CoolCellMap"],
           ["Hcal/PedestalMonitor/HE/HE Normalized RMS Values",
            "Hcal/PedestalMonitor/HE/HE Subtracted Mean Values",
            "Hcal/PedestalMonitor/HE/HE CapID RMS Variance",
            "Hcal/PedestalMonitor/HE/HE CapID Mean Variance"],
           ["Hcal/LEDMonitor/HE/HE Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HE/HE Average Pulse Shape",
            "Hcal/LEDMonitor/HE/HE Average Pulse Time"]
)

hcallayout(dqmitems, "HCAL Forward Summary",
           ["Hcal/DigiMonitor/HF/HF # of Digis",
            "Hcal/DigiMonitor/HF/HF QIE ADC Value"],
           ["Hcal/RecHitMonitor/HF/HF RecHit Energies",
            "Hcal/RecHitMonitor/HF/HF RecHit Times",
            "Hcal/RecHitMonitor/HF/HF RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HF/HF RecHit Times - Threshold"],
           ["Hcal/HotCellMonitor/HF/HFHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HF/NADA_HF_OCC_MAP"],
           ["Hcal/DeadCellMonitor/HF/HF_deadADCOccupancyMap",
            "Hcal/DeadCellMonitor/HF/HF_CoolCell_belowPed",
            "Hcal/DeadCellMonitor/HF/HF_NADA_CoolCellMap"],
           ["Hcal/PedestalMonitor/HF/HF Normalized RMS Values",
            "Hcal/PedestalMonitor/HF/HF Subtracted Mean Values",
            "Hcal/PedestalMonitor/HF/HF CapID RMS Variance",
            "Hcal/PedestalMonitor/HF/HF CapID Mean Variance"],
           ["Hcal/LEDMonitor/HF/HF Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HF/HF Average Pulse Shape",
            "Hcal/LEDMonitor/HF/HF Average Pulse Time"]

)

hcallayout(dqmitems, "HCAL Outer Summary",
           ["Hcal/DigiMonitor/HO/HO # of Digis",
            "Hcal/DigiMonitor/HO/HO QIE ADC Value"],
           ["Hcal/RecHitMonitor/HO/HO RecHit Energies",
            "Hcal/RecHitMonitor/HO/HO RecHit Times",
            "Hcal/RecHitMonitor/HO/HO RecHit Total Energy - Threshold",
            "Hcal/RecHitMonitor/HO/HO RecHit Times - Threshold"],
           ["Hcal/HotCellMonitor/HO/HOHotCellOCCmap_Thresh0",
            "Hcal/HotCellMonitor/HO/NADA_HO_OCC_MAP"],
           ["Hcal/DeadCellMonitor/HO/HO_deadADCOccupancyMap",
            "Hcal/DeadCellMonitor/HO/HO_CoolCell_belowPed",
            "Hcal/DeadCellMonitor/HO/HO_NADA_CoolCellMap"],
           ["Hcal/PedestalMonitor/HO/HO Normalized RMS Values",
            "Hcal/PedestalMonitor/HO/HO Subtracted Mean Values",
            "Hcal/PedestalMonitor/HO/HO CapID RMS Variance",
            "Hcal/PedestalMonitor/HO/HO CapID Mean Variance"],
           ["Hcal/LEDMonitor/HO/HO Ped Subtracted Pulse Shape",
            "Hcal/LEDMonitor/HO/HO Average Pulse Shape",
            "Hcal/LEDMonitor/HO/HO Average Pulse Time"]
)
