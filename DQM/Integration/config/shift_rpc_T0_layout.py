def shiftrpclayout(i, p, *rows): i["00 Shift/RPC/" + p] = DQMItem(layout=rows)

summary = "summary map for rpc, this is NOT an efficiency measurement"
fed = "FED Fatal Errors. Entries MUST be ZERO at all times. If not, report the problem and ask to stop the run."
rpcevents = "Number of processed events."
eff = "RPC Efficiency distribution. Make sure average values is greater than 80."
rpclink = "   >>> <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftRPC>Description</a>"
quality = "Overview of system quality. Expressed in percentage of chambers."
occupancy = "Occupancy per sector"
hv = "High Voltage status per lumi section. If ALL bins are red mark the run as BAD."

shiftrpclayout(dqmitems, "00-Summary_Map",
               [{ 'path': "RPC/EventInfo/reportSummaryMap", 'description': summary + rpclink }])

shiftrpclayout(dqmitems, "01-FED_Fatal_Errors",
               [{ 'path': "RPC/FEDIntegrity/FEDFatal", 'description': fed + rpclink }])
          
shiftrpclayout(dqmitems, "02-RPC_HV_Status",
               [{ 'path': "RPC/DCSInfo/rpcHVStatus", 'description': hv + rpclink }])


shiftrpclayout(dqmitems, "03-RPC_Events",
               [{ 'path': "RPC/AllHits/RPCEvents", 'description': rpcevents + rpclink }])


shiftrpclayout(dqmitems, "04-Quality_State_Overview",
               [{ 'path': "RPC/AllHits/SummaryHistograms/RPC_System_Quality_Overview", 'description': quality + rpclink }])


shiftrpclayout(dqmitems, "05-RPC_Occupancy",
               [{ 'path': "RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel", 'description': occupancy + rpclink  }],
               [{ 'path': "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap", 'description': occupancy + rpclink  }])


shiftrpclayout(dqmitems, "06-Statistics",
               [{ 'path': "RPC/RPCEfficiency/Statistics", 'description': eff + rpclink }])


shiftrpclayout(dqmitems, "07-Efficiency_Distribution",
               [{ 'path': "RPC/RPCEfficiency/EffBarrelRoll", 'description': eff + rpclink  }],
               [{ 'path': "RPC/RPCEfficiency/EffEndcapPlusRoll", 'description': eff + rpclink  },
                { 'path': "RPC/RPCEfficiency/EffEndcapMinusRoll", 'description':  eff + rpclink }])
