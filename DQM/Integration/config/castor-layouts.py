def castorlayout(i, p, *rows): i["Castor/Layouts/" + p] = DQMItem(layout=rows)


castorlayout(dqmitems, "CASTOR All Digi Values",
           [{ 'path': "Castor/CastorDigiMonitor/Castor All Digi Values",
             'description':"all CASTOR ADC values"}]
           )

castorlayout(dqmitems, "CASTOR Digi ChannelSummaryMap",
           [{ 'path': "Castor/CastorPSMonitor/CASTOR Digi ChannelSummaryMap",
             'description':""}]
           )

castorlayout(dqmitems, "CASTOR Digi Occupancy Map",
           [{ 'path': "Castor/CastorPSMonitor/CASTOR Digi Occupancy Map",
             'description':"dynamic scale"}]
           )

castorlayout(dqmitems, "CASTOR Digi SaturationSummaryMap",
           [{ 'path': "Castor/CastorPSMonitor/CASTOR Digi SaturationSummaryMap",
             'description':""}]
           )

castorlayout(dqmitems, "CASTOR event products",
           [{ 'path': "Castor/CastorEventProducts/CastorEventProduct",
             'description':"check whether CASTOR objects are present in the events"}]
           )

castorlayout(dqmitems, "CASTOR RecHit Occupancy Map",
           [{ 'path': "Castor/CastorRecHitMonitor/CastorRecHits Occupancy Map",
             'description':""}]
           )

castorlayout(dqmitems, "CASTOR RecHit Energy Fraction in modules",
           [{ 'path': "Castor/CastorRecHitMonitor/EnergyFraction/Fraction of the total energy in CASTOR modules",
             'description':""}]
           )

castorlayout(dqmitems, "CASTOR RecHit Energy Fraction in sectors",
           [{ 'path': "Castor/CastorRecHitMonitor/EnergyFraction/Fraction of the total energy in CASTOR sectors",
             'description':""}]
           )

castorlayout(dqmitems, "CASTOR RecHit number per event- above threshold",
           [{ 'path': "Castor/CastorRecHitMonitor/Number of CASTOR RecHits per event- above threshold",
             'description':""}]
           )
              
castorlayout(dqmitems, "CASTOR RecHit Energies",
           [{ 'path': "Castor/CastorRecHitMonitor/CastorRecHit Energies- above threshold on RecHitEnergy",
             'description':"Energy of all Castor RecHits"}]
           )
	  
castorlayout(dqmitems, "CASTOR RecHit Energy in modules",
           [{ 'path': "Castor/CastorRecHitMonitor/CastorRecHit Energy in modules- above threshold",
             'description':"RecHitEnergy in each of 14 CASTOR modules"}]
           )          
	 
castorlayout(dqmitems, "CASTOR RecHit Energy in sectors",
           [{ 'path': "Castor/CastorRecHitMonitor/CastorRecHit Energy in sectors- above threshold",
             'description':"RecHitEnergy in each of 16 CASTOR sectors"}]
           )         
	  
castorlayout(dqmitems, "CASTOR RecHit Energy 2D Map",
           [{ 'path': "Castor/CastorRecHitMonitor/CastorRecHit 2D Energy Map- above threshold",
             'description':"2D Energy Map"}]
           )
	  	  

castorlayout(dqmitems, "CASTOR hits 3D- cumulative",
           [{ 'path': "Castor/CastorEventDisplay/CASTOR 3D hits- cumulative",
             'description':"cumulative event display"}]
           )         

castorlayout(dqmitems, "CASTOR hits 3D- event with the largest deposited E",
           [{ 'path': "Castor/CastorEventDisplay/CASTOR 3D hits- event with the largest deposited E",
             'description':"display of the event with largest deposited energy"}]
           )


castorlayout(dqmitems, "CASTOR RecHit Energy per event",
           [{ 'path': "Castor/CastorHIMonitor/EnergyUnits/CASTOR Absolute RecHit Energy per event",
             'description':"total energy in CASTOR per event - sum over all 224 channels"}]
           )

castorlayout(dqmitems, "CASTOR Total RecHit Energy in phi-sectors per run",
           [{ 'path': "Castor/CastorHIMonitor/EnergyUnits/CASTOR Total RecHit Energy in phi-sectors per run",
             'description':" total energy in each CASTOR phi-sector: energy vs phi-sector"}]
           )

castorlayout(dqmitems, "CASTOR Total EM RecHit Energy per event",
           [{ 'path': "Castor/CastorHIMonitor/EnergyUnits/CASTOR Total EM RecHit Energy per event",
             'description':"total EM energy per event"}]
           )

castorlayout(dqmitems, "CASTOR Total HAD RecHit Energy per event",
           [{ 'path': "Castor/CastorHIMonitor/EnergyUnits/CASTOR Total HAD RecHit Energy per event",
             'description':"total HAD energy per event"}]
           )

castorlayout(dqmitems, "CASTOR Total Energy ratio EM to HAD per event",
           [{ 'path': "Castor/CastorHIMonitor/EnergyUnits/CASTOR Total Energy ratio EM to HAD per event",
             'description':"total energy ratio EM to HAD per event"}]
           )

castorlayout(dqmitems, "CASTOR average pulse in bunch crossings",
           [{ 'path': "Castor/CastorPSMonitor/CASTOR average pulse in bunch crossings",
             'description':"average pulse in bunch crossings"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=1 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=1 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=2 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=2 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=3 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=3 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=4 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=4 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=5 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=5 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=6 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=6 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           )   

castorlayout(dqmitems, "Castor Pulse Shape for sector=7 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=7 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=8 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=8 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=9 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=9 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=10 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=10 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=11 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=11 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=12 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=12 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=13 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=13 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=14 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=14 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=15 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=15 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 

castorlayout(dqmitems, "Castor Pulse Shape for sector=16 (in all 14 modules)",
           [{ 'path': "Castor/CastorPSMonitor/Castor Pulse Shape for sector=16 (in all 14 modules)",
             'description':"pulse shape in this particular sector"}]
           ) 
