
ALTER TABLE  CALI_HV_SCAN_RATIO_DAT          SPLIT PARTITION                    
CALI_HV_SCAN_RATIO_DAT_09                                                       
 AT (          0 ) INTO (PARTITION                                              
CALI_HV_SCAN_RATIO_DAT_10                                                       
, PARTITION                                                                     
CALI_HV_SCAN_RATIO_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  CALI_TEMP_DAT                   SPLIT PARTITION                    
CALI_TEMP_DAT_09                                                                
 AT (        143 ) INTO (PARTITION                                              
CALI_TEMP_DAT_10                                                                
, PARTITION                                                                     
CALI_TEMP_DAT_11                                                                
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_IDARK_DAT                   SPLIT PARTITION                    
DCU_IDARK_DAT_0                                                                
 AT (      41694 ) INTO (PARTITION                                              
DCU_IDARK_DAT_10                                                                
, PARTITION                                                                     
DCU_IDARK_DAT_11                                                                
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_BLUE_PN_PRIM_DAT      SPLIT PARTITION                    
LMF_LASER_BLUE_PN_PRIM_DAT_09                                                   
 AT (     223235 ) INTO (PARTITION                                              
LMF_LASER_BLUE_PN_PRIM_DAT_10                                                   
, PARTITION                                                                     
LMF_LASER_BLUE_PN_PRIM_DAT_11                                                   
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_IRED_PRIM_DAT         SPLIT PARTITION                    
LMF_LASER_IRED_PRIM_DAT_09                                                      
 AT (          0 ) INTO (PARTITION                                              
LMF_LASER_IRED_PRIM_DAT_10                                                      
, PARTITION                                                                     
LMF_LASER_IRED_PRIM_DAT_11                                                      
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_H4_TABLE_POSITION_DAT       SPLIT PARTITION                    
MON_H4_TABLE_POSITION_DAT_09                                                    
 AT (          0 ) INTO (PARTITION                                              
MON_H4_TABLE_POSITION_DAT_10                                                    
, PARTITION                                                                     
MON_H4_TABLE_POSITION_DAT_11                                                    
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LASER_IRED_DAT              SPLIT PARTITION                    
MON_LASER_IRED_DAT_09                                                           
 AT (       3001 ) INTO (PARTITION                                              
MON_LASER_IRED_DAT_10                                                           
, PARTITION                                                                     
MON_LASER_IRED_DAT_11                                                           
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_SHAPE_QUALITY_DAT           SPLIT PARTITION                    
MON_SHAPE_QUALITY_DAT_09                                                        
 AT (          0 ) INTO (PARTITION                                              
MON_SHAPE_QUALITY_DAT_10                                                        
, PARTITION                                                                     
MON_SHAPE_QUALITY_DAT_11                                                        
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_CRYSTAL_DAT          SPLIT PARTITION                    
MON_TIMING_CRYSTAL_DAT_0                                                       
 AT (      50314 ) INTO (PARTITION                                              
MON_TIMING_CRYSTAL_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_CRYSTAL_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_XTAL_LI_DAT          SPLIT PARTITION                    
MON_TIMING_XTAL_LI_DAT_09                                                       
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_XTAL_LI_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_XTAL_LI_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  OD_CCS_HF_DAT                   SPLIT PARTITION                    
OD_CCS_HF_DAT_09                                                                
 AT (          0 ) INTO (PARTITION                                              
OD_CCS_HF_DAT_10                                                                
, PARTITION                                                                     
OD_CCS_HF_DAT_11                                                                
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  OD_CCS_TR_DAT                   SPLIT PARTITION                    
OD_CCS_TR_DAT_09                                                                
 AT (          0 ) INTO (PARTITION                                              
OD_CCS_TR_DAT_10                                                                
, PARTITION                                                                     
OD_CCS_TR_DAT_11                                                                
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  OD_DCC_DETAILS_DAT              SPLIT PARTITION                    
OD_DCC_DETAILS_DAT_0                                                           
 AT (      12265 ) INTO (PARTITION                                              
OD_DCC_DETAILS_DAT_10                                                           
, PARTITION                                                                     
OD_DCC_DETAILS_DAT_11                                                           
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_CRYSTAL_ERRORS_DAT          SPLIT PARTITION                    
RUN_CRYSTAL_ERRORS_DAT_0                                                     
 AT (      47094 ) INTO (PARTITION                                              
RUN_CRYSTAL_ERRORS_DAT_10                                                       
, PARTITION                                                                     
RUN_CRYSTAL_ERRORS_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_IOV                         SPLIT PARTITION                    
RUN_IOV_0                                                                      
 AT (      50960 ) INTO (PARTITION                                              
RUN_IOV_10                                                                      
, PARTITION                                                                     
RUN_IOV_11                                                                      
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_MEM_TT_ERRORS_DAT           SPLIT PARTITION                    
RUN_MEM_TT_ERRORS_DAT_0                                                        
 AT (      47094 ) INTO (PARTITION                                              
RUN_MEM_TT_ERRORS_DAT_10                                                        
, PARTITION                                                                     
RUN_MEM_TT_ERRORS_DAT_11                                                        
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  CALI_IOV                        SPLIT PARTITION                    
CALI_IOV_09                                                                     
 AT (        143 ) INTO (PARTITION                                              
CALI_IOV_10                                                                     
, PARTITION                                                                     
CALI_IOV_11                                                                     
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_IDARK_PED_DAT               SPLIT PARTITION                    
DCU_IDARK_PED_DAT_09                                                            
 AT (      24070 ) INTO (PARTITION                                              
DCU_IDARK_PED_DAT_10                                                            
, PARTITION                                                                     
DCU_IDARK_PED_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_BLUE_PRIM_DAT         SPLIT PARTITION                    
LMF_LASER_BLUE_PRIM_DAT_09                                                      
 AT (     223235 ) INTO (PARTITION                                              
LMF_LASER_BLUE_PRIM_DAT_10                                                      
, PARTITION                                                                     
LMF_LASER_BLUE_PRIM_DAT_11                                                      
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_IRED_PN_PRIM_DAT      SPLIT PARTITION                    
LMF_LASER_IRED_PN_PRIM_DAT_09                                                   
 AT (          0 ) INTO (PARTITION                                              
LMF_LASER_IRED_PN_PRIM_DAT_10                                                   
, PARTITION                                                                     
LMF_LASER_IRED_PN_PRIM_DAT_11                                                   
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_TEST_PULSE_CONFIG_DAT       SPLIT PARTITION                    
LMF_TEST_PULSE_CONFIG_DAT_09                                                    
 AT (          0 ) INTO (PARTITION                                              
LMF_TEST_PULSE_CONFIG_DAT_10                                                    
, PARTITION                                                                     
LMF_TEST_PULSE_CONFIG_DAT_11                                                    
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LED1_DAT                    SPLIT PARTITION                    
MON_LED1_DAT_0                                                                 
 AT (      50313 ) INTO (PARTITION                                              
MON_LED1_DAT_10                                                                 
, PARTITION                                                                     
MON_LED1_DAT_11                                                                 
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_OCCUPANCY_DAT               SPLIT PARTITION                    
MON_OCCUPANCY_DAT_0                                                            
 AT (      50314 ) INTO (PARTITION                                              
MON_OCCUPANCY_DAT_10                                                            
, PARTITION                                                                     
MON_OCCUPANCY_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PEDESTAL_OFFSETS_DAT        SPLIT PARTITION                    
MON_PEDESTAL_OFFSETS_DAT_09                                                     
 AT (      28754 ) INTO (PARTITION                                              
MON_PEDESTAL_OFFSETS_DAT_10                                                     
, PARTITION                                                                     
MON_PEDESTAL_OFFSETS_DAT_11                                                     
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TEST_PULSE_DAT              SPLIT PARTITION                    
MON_TEST_PULSE_DAT_0                                                          
 AT (      50313 ) INTO (PARTITION                                              
MON_TEST_PULSE_DAT_10                                                           
, PARTITION                                                                     
MON_TEST_PULSE_DAT_11                                                           
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_L1_DAT            SPLIT PARTITION                    
MON_TIMING_TT_L1_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_L1_DAT_10                                                         
, PARTITION                                                                     
MON_TIMING_TT_L1_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_LI_DAT            SPLIT PARTITION                    
MON_TIMING_TT_LI_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_LI_DAT_10                                                         
, PARTITION                                                                     
MON_TIMING_TT_LI_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  OD_CCS_FE_DAT                   SPLIT PARTITION                    
OD_CCS_FE_DAT_0                                                                
 AT (      12265 ) INTO (PARTITION                                              
OD_CCS_FE_DAT_10                                                                
, PARTITION                                                                     
OD_CCS_FE_DAT_11                                                                
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  OD_DCC_OPERATION_DAT            SPLIT PARTITION                    
OD_DCC_OPERATION_DAT_0                                                         
 AT (      12265 ) INTO (PARTITION                                              
OD_DCC_OPERATION_DAT_10                                                         
, PARTITION                                                                     
OD_DCC_OPERATION_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  OD_RUN_IOV                      SPLIT PARTITION                    
OD_RUN_IOV_09                                                                   
 AT (      12265 ) INTO (PARTITION                                              
OD_RUN_IOV_10                                                                   
, PARTITION                                                                     
OD_RUN_IOV_11                                                                   
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_H4_TABLE_POSITION_DAT       SPLIT PARTITION                    
RUN_H4_TABLE_POSITION_DAT_0                                                    
 AT (          0 ) INTO (PARTITION                                              
RUN_H4_TABLE_POSITION_DAT_10                                                    
, PARTITION                                                                     
RUN_H4_TABLE_POSITION_DAT_11                                                    
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_PN_ERRORS_DAT               SPLIT PARTITION                    
RUN_PN_ERRORS_DAT_0                                                            
 AT (      47094 ) INTO (PARTITION                                              
RUN_PN_ERRORS_DAT_10                                                            
, PARTITION                                                                     
RUN_PN_ERRORS_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  CALI_CRYSTAL_INTERCAL_DAT       SPLIT PARTITION                    
CALI_CRYSTAL_INTERCAL_DAT_09                                                    
 AT (          5 ) INTO (PARTITION                                              
CALI_CRYSTAL_INTERCAL_DAT_10                                                    
, PARTITION                                                                     
CALI_CRYSTAL_INTERCAL_DAT_11                                                    
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_LVRB_TEMPS_DAT              SPLIT PARTITION                    
DCU_LVRB_TEMPS_DAT_0                                                           
 AT (      41694 ) INTO (PARTITION                                              
DCU_LVRB_TEMPS_DAT_10                                                           
, PARTITION                                                                     
DCU_LVRB_TEMPS_DAT_11                                                           
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_VFE_TEMP_DAT                SPLIT PARTITION                    
DCU_VFE_TEMP_DAT_0                                                             
 AT (      41694 ) INTO (PARTITION                                              
DCU_VFE_TEMP_DAT_10                                                             
, PARTITION                                                                     
DCU_VFE_TEMP_DAT_11                                                             
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_IRED_PULSE_DAT        SPLIT PARTITION                    
LMF_LASER_IRED_PULSE_DAT_09                                                     
 AT (          0 ) INTO (PARTITION                                              
LMF_LASER_IRED_PULSE_DAT_10                                                     
, PARTITION                                                                     
LMF_LASER_IRED_PULSE_DAT_11                                                     
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LED2_DAT                    SPLIT PARTITION                    
MON_LED2_DAT_0                                                                 
 AT (      50313 ) INTO (PARTITION                                              
MON_LED2_DAT_10                                                                 
, PARTITION                                                                     
MON_LED2_DAT_11                                                                 
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PEDESTALS_DAT               SPLIT PARTITION                    
MON_PEDESTALS_DAT_0                                                            
 AT (      50314 ) INTO (PARTITION                                              
MON_PEDESTALS_DAT_10                                                            
, PARTITION                                                                     
MON_PEDESTALS_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_BLUE_DAT                 SPLIT PARTITION                    
MON_PN_BLUE_DAT_0                                                              
 AT (      50313 ) INTO (PARTITION                                              
MON_PN_BLUE_DAT_10                                                              
, PARTITION                                                                     
MON_PN_BLUE_DAT_11                                                              
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_STATUS_DAT               SPLIT PARTITION                    
MON_PN_STATUS_DAT_09                                                            
 AT (          0 ) INTO (PARTITION                                              
MON_PN_STATUS_DAT_10                                                            
, PARTITION                                                                     
MON_PN_STATUS_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_RUN_DAT                     SPLIT PARTITION                    
MON_RUN_DAT_0                                                                  
 AT (      50314 ) INTO (PARTITION                                              
MON_RUN_DAT_10                                                                  
, PARTITION                                                                     
MON_RUN_DAT_11                                                                  
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_LG_DAT            SPLIT PARTITION                    
MON_TIMING_TT_LG_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_LG_DAT_10                                                         
, PARTITION                                                                     
MON_TIMING_TT_LG_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_XTAL_L1_DAT          SPLIT PARTITION                    
MON_TIMING_XTAL_L1_DAT_0                                                       
 AT (      50313 ) INTO (PARTITION                                              
MON_TIMING_XTAL_L1_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_XTAL_L1_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_XTAL_L2_DAT          SPLIT PARTITION                    
MON_TIMING_XTAL_L2_DAT_0                                                       
 AT (      50313 ) INTO (PARTITION                                              
MON_TIMING_XTAL_L2_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_XTAL_L2_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_FECONFIG_DAT                SPLIT PARTITION                    
RUN_FECONFIG_DAT_0                                                             
 AT (      50960 ) INTO (PARTITION                                              
RUN_FECONFIG_DAT_10                                                             
, PARTITION                                                                     
RUN_FECONFIG_DAT_11                                                             
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_TT_ERRORS_DAT               SPLIT PARTITION                    
RUN_TT_ERRORS_DAT_0                                                            
 AT (      47094 ) INTO (PARTITION                                              
RUN_TT_ERRORS_DAT_10                                                            
, PARTITION                                                                     
RUN_TT_ERRORS_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_CTRLBOX_MON           SPLIT PARTITION                    
LED_PWRCTRLBX_M_0                                                               
 AT (     123551 ) INTO (PARTITION                                              
LED_PWRCTRLBX_M_1                                                               
, PARTITION                                                                     
LED_PWRCTRLBX_M_2                                                               
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_SUPPLY_MON            SPLIT PARTITION                    
LED_PWRSPPL_M_0                                                                 
 AT (     123551 ) INTO (PARTITION                                              
LED_PWRSPPL_M_1                                                                 
, PARTITION                                                                     
LED_PWRSPPL_M_2                                                                 
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_CALIB_PRIM_DAT              SPLIT PARTITION                    
LMF_CALIB_PRIM_DAT_09                                                           
 AT (          0 ) INTO (PARTITION                                              
LMF_CALIB_PRIM_DAT_10                                                           
, PARTITION                                                                     
LMF_CALIB_PRIM_DAT_11                                                           
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LASER_RED_DAT               SPLIT PARTITION                    
MON_LASER_RED_DAT_0                                                            
 AT (      50314 ) INTO (PARTITION                                              
MON_LASER_RED_DAT_10                                                            
, PARTITION                                                                     
MON_LASER_RED_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_PED_DAT                  SPLIT PARTITION                    
MON_PN_PED_DAT_0                                                               
 AT (      50314 ) INTO (PARTITION                                              
MON_PN_PED_DAT_10                                                               
, PARTITION                                                                     
MON_PN_PED_DAT_11                                                               
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_IOV                         SPLIT PARTITION                    
DCU_IOV_0                                                                      
 AT (      41694 ) INTO (PARTITION                                              
DCU_IOV_10                                                                      
, PARTITION                                                                     
DCU_IOV_11                                                                      
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_DELAYS_TT_DAT               SPLIT PARTITION                    
MON_DELAYS_TT_DAT_09                                                            
 AT (          0 ) INTO (PARTITION                                              
MON_DELAYS_TT_DAT_10                                                            
, PARTITION                                                                     
MON_DELAYS_TT_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_MEM_CH_CONSISTENCY_DAT      SPLIT PARTITION                    
MON_MEM_CH_CONSISTENCY_DAT_0                                                   
 AT (      47497 ) INTO (PARTITION                                              
MON_MEM_CH_CONSISTENCY_DAT_10                                                   
, PARTITION                                                                     
MON_MEM_CH_CONSISTENCY_DAT_11                                                   
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_GREEN_DAT                SPLIT PARTITION                    
MON_PN_GREEN_DAT_09                                                             
 AT (          0 ) INTO (PARTITION                                              
MON_PN_GREEN_DAT_10                                                             
, PARTITION                                                                     
MON_PN_GREEN_DAT_11                                                             
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PULSE_SHAPE_DAT             SPLIT PARTITION                    
MON_PULSE_SHAPE_DAT_0                                                          
 AT (      50313 ) INTO (PARTITION                                              
MON_PULSE_SHAPE_DAT_10                                                          
, PARTITION                                                                     
MON_PULSE_SHAPE_DAT_11                                                          
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_DAT               SPLIT PARTITION                    
MON_TIMING_TT_DAT_09                                                            
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_DAT_10                                                            
, PARTITION                                                                     
MON_TIMING_TT_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_L2_DAT            SPLIT PARTITION                    
MON_TIMING_TT_L2_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_L2_DAT_10                                                         
, PARTITION                                                                     
MON_TIMING_TT_L2_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_LB_DAT            SPLIT PARTITION                    
MON_TIMING_TT_LB_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_LB_DAT_10                                                         
, PARTITION                                                                     
MON_TIMING_TT_LB_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_DAT                         SPLIT PARTITION                    
RUN_DAT_0                                                                      
 AT (      50960 ) INTO (PARTITION                                              
RUN_DAT_10                                                                      
, PARTITION                                                                     
RUN_DAT_11                                                                      
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_MEM_CH_ERRORS_DAT           SPLIT PARTITION                    
RUN_MEM_CH_ERRORS_DAT_09                                                        
 AT (          0 ) INTO (PARTITION                                              
RUN_MEM_CH_ERRORS_DAT_10                                                        
, PARTITION                                                                     
RUN_MEM_CH_ERRORS_DAT_11                                                        
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_TPGCONFIG_DAT               SPLIT PARTITION                    
RUN_TPGCONFIG_DAT_0                                                            
 AT (      50960 ) INTO (PARTITION                                              
RUN_TPGCONFIG_DAT_10                                                            
, PARTITION                                                                     
RUN_TPGCONFIG_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_CAPSULE_TEMP_RAW_DAT        SPLIT PARTITION                    
DCU_CAPSULE_TEMP_RAW_DAT_0                                                     
 AT (      31402 ) INTO (PARTITION                                              
DCU_CAPSULE_TEMP_RAW_DAT_10                                                     
, PARTITION                                                                     
DCU_CAPSULE_TEMP_RAW_DAT_11                                                     
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_CTRLBOX_STATUS        SPLIT PARTITION                    
LED_PWRCTRLBX_S_0                                                               
 AT (     123552 ) INTO (PARTITION                                              
LED_PWRCTRLBX_S_1                                                               
, PARTITION                                                                     
LED_PWRCTRLBX_S_2                                                               
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_SUPPLY_STATUS         SPLIT PARTITION                    
LED_PWRSPPL_S_0                                                                 
 AT (     123552 ) INTO (PARTITION                                              
LED_PWRSPPL_S_1                                                                 
, PARTITION                                                                     
LED_PWRSPPL_S_2                                                                 
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_RUN_DAT                     SPLIT PARTITION                    
LMF_RUN_DAT_09                                                                  
 AT (     223235 ) INTO (PARTITION                                              
LMF_RUN_DAT_10                                                                  
, PARTITION                                                                     
LMF_RUN_DAT_11                                                                  
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_RUN_IOV                     SPLIT PARTITION                    
LMF_RUN_IOV_0                                                                  
 AT (     223235 ) INTO (PARTITION                                              
LMF_RUN_IOV_10                                                                  
, PARTITION                                                                     
LMF_RUN_IOV_11                                                                  
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_CRYSTAL_CONSISTENCY_DAT     SPLIT PARTITION                    
MON_CRYSTAL_CONSISTENCY_DAT_09                                                  
 AT (          0 ) INTO (PARTITION                                              
MON_CRYSTAL_CONSISTENCY_DAT_10                                                  
, PARTITION                                                                     
MON_CRYSTAL_CONSISTENCY_DAT_11                                                  
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_CRYSTAL_STATUS_DAT          SPLIT PARTITION                    
MON_CRYSTAL_STATUS_DAT_09                                                       
 AT (          0 ) INTO (PARTITION                                              
MON_CRYSTAL_STATUS_DAT_10                                                       
, PARTITION                                                                     
MON_CRYSTAL_STATUS_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  CALI_GAIN_RATIO_DAT             SPLIT PARTITION                    
CALI_GAIN_RATIO_DAT_09                                                          
 AT (          0 ) INTO (PARTITION                                              
CALI_GAIN_RATIO_DAT_10                                                          
, PARTITION                                                                     
CALI_GAIN_RATIO_DAT_11                                                          
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_LED_MON               SPLIT PARTITION                    
LED_PWRLD_M_0                                                                   
 AT (     123551 ) INTO (PARTITION                                              
LED_PWRLD_M_1                                                                   
, PARTITION                                                                     
LED_PWRLD_M_2                                                                   
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_MONBOX_STATUS         SPLIT PARTITION                    
LED_PWRMNBX_S_0                                                                 
 AT (     123552 ) INTO (PARTITION                                              
LED_PWRMNBX_S_1                                                                 
, PARTITION                                                                     
LED_PWRMNBX_S_2                                                                 
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_SYSTEM_IOV                  SPLIT PARTITION                    
LED_SYSTEM_IOV_0                                                                
 AT (     123552 ) INTO (PARTITION                                              
LED_SYSTEM_IOV_1                                                                
, PARTITION                                                                     
LED_SYSTEM_IOV_2                                                                
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_CONFIG_DAT            SPLIT PARTITION                    
LMF_LASER_CONFIG_DAT_09                                                         
 AT (     223235 ) INTO (PARTITION                                              
LMF_LASER_CONFIG_DAT_10                                                         
, PARTITION                                                                     
LMF_LASER_CONFIG_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LASER_GREEN_DAT             SPLIT PARTITION                    
MON_LASER_GREEN_DAT_09                                                          
 AT (          0 ) INTO (PARTITION                                              
MON_LASER_GREEN_DAT_10                                                          
, PARTITION                                                                     
MON_LASER_GREEN_DAT_11                                                          

 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_IRED_DAT                 SPLIT PARTITION                    
MON_PN_IRED_DAT_09                                                              
 AT (       3001 ) INTO (PARTITION                                              
MON_PN_IRED_DAT_10                                                              
, PARTITION                                                                     
MON_PN_IRED_DAT_11                                                              
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_MGPA_DAT                 SPLIT PARTITION                    
MON_PN_MGPA_DAT_0                                                              
 AT (      50313 ) INTO (PARTITION                                              
MON_PN_MGPA_DAT_10                                                              
, PARTITION                                                                     
MON_PN_MGPA_DAT_11                                                              
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_RUN_IOV                     SPLIT PARTITION                    
MON_RUN_IOV_0                                                               
 AT (      50314 ) INTO (PARTITION                                              
MON_RUN_IOV_10                                                                  
, PARTITION                                                                     
MON_RUN_IOV_11                                                                  
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_XTAL_LR_DAT          SPLIT PARTITION                    
MON_TIMING_XTAL_LR_DAT_0                                                       
 AT (      50314 ) INTO                                               
MON_TIMING_XTAL_LR_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_XTAL_LR_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  CALI_GENERAL_DAT                SPLIT PARTITION                    
CALI_GENERAL_DAT_09                                                             
 AT (          0 ) INTO (PARTITION                                              
CALI_GENERAL_DAT_10                                                             
, PARTITION                                                                     
CALI_GENERAL_DAT_11                                                             
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_CAPSULE_TEMP_DAT            SPLIT PARTITION                    
DCU_CAPSULE_TEMP_DAT_09                                                         
 AT (      41694 ) INTO (PARTITION                                              
DCU_CAPSULE_TEMP_DAT_10                                                         
, PARTITION                                                                     
DCU_CAPSULE_TEMP_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_LVR_TEMPS_DAT               SPLIT PARTITION                    
DCU_LVR_TEMPS_DAT_0                                                            
 AT (      41694 ) INTO (PARTITION                                              
DCU_LVR_TEMPS_DAT_10                                                            
, PARTITION                                                                     
DCU_LVR_TEMPS_DAT_11                                                            
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  DCU_LVR_VOLTAGES_DAT            SPLIT PARTITION                    
DCU_LVR_VOLTAGES_DAT_0                                                         
 AT (      41694 ) INTO (PARTITION                                              
DCU_LVR_VOLTAGES_DAT_10                                                         
, PARTITION                                                                     
DCU_LVR_VOLTAGES_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LED_POWER_MONBOX_MON            SPLIT PARTITION                    
LED_PWRMNBX_M_0                                                                 
 AT (     123551 ) INTO (PARTITION                                              
LED_PWRMNBX_M_1                                                                 
, PARTITION                                                                     
LED_PWRMNBX_M_2                                                                 
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  LMF_LASER_BLUE_PULSE_DAT        SPLIT PARTITION                    
LMF_LASER_BLUE_PULSE_DAT_09                                                     
 AT (     223235 ) INTO (PARTITION                                              
LMF_LASER_BLUE_PULSE_DAT_10                                                     
, PARTITION                                                                     
LMF_LASER_BLUE_PULSE_DAT_11                                                     
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LASER_BLUE_DAT              SPLIT PARTITION                    
MON_LASER_BLUE_DAT_0                                                           
 AT (      50313 ) INTO (PARTITION                                              
MON_LASER_BLUE_DAT_10                                                           
, PARTITION                                                                     
MON_LASER_BLUE_DAT_11                                                           
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LASER_PULSE_DAT             SPLIT PARTITION                    
MON_LASER_PULSE_DAT_09                                                          
 AT (          0 ) INTO (PARTITION                                              
MON_LASER_PULSE_DAT_10                                                          
, PARTITION                                                                     
MON_LASER_PULSE_DAT_11                                                          
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_LASER_STATUS_DAT            SPLIT PARTITION                    
MON_LASER_STATUS_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_LASER_STATUS_DAT_10                                                         
, PARTITION                                                                     
MON_LASER_STATUS_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_MEM_TT_CONSISTENCY_DAT      SPLIT PARTITION                    
MON_MEM_TT_CONSISTENCY_DAT_0                                                   
 AT (      49184 ) INTO (PARTITION                                              
MON_MEM_TT_CONSISTENCY_DAT_10                                                   
, PARTITION                                                                     
MON_MEM_TT_CONSISTENCY_DAT_11                                                   
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PEDESTALS_ONLINE_DAT        SPLIT PARTITION                    
MON_PEDESTALS_ONLINE_DAT_0                                                     
 AT (      50314 ) INTO (PARTITION                                              
MON_PEDESTALS_ONLINE_DAT_10                                                     
, PARTITION                                                                     
MON_PEDESTALS_ONLINE_DAT_11                                                     
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_PN_RED_DAT                  SPLIT PARTITION                    
MON_PN_RED_DAT_0                                                               
 AT (      50314 ) INTO (PARTITION                                              
MON_PN_RED_DAT_10                                                               
, PARTITION                                                                     
MON_PN_RED_DAT_11                                                               
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_TT_LR_DAT            SPLIT PARTITION                    
MON_TIMING_TT_LR_DAT_09                                                         
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_TT_LR_DAT_10                                                         
, PARTITION                                                                     
MON_TIMING_TT_LR_DAT_11                                                         
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_XTAL_LB_DAT          SPLIT PARTITION                    
MON_TIMING_XTAL_LB_DAT_0                                                       
 AT (      50313 ) INTO (PARTITION                                              
MON_TIMING_XTAL_LB_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_XTAL_LB_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TIMING_XTAL_LG_DAT          SPLIT PARTITION                    
MON_TIMING_XTAL_LG_DAT_09                                                       
 AT (          0 ) INTO (PARTITION                                              
MON_TIMING_XTAL_LG_DAT_10                                                       
, PARTITION                                                                     
MON_TIMING_XTAL_LG_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  MON_TT_CONSISTENCY_DAT          SPLIT PARTITION                    
MON_TT_CONSISTENCY_DAT_0                                                       
 AT (      49772 ) INTO (PARTITION                                              
MON_TT_CONSISTENCY_DAT_10                                                       
, PARTITION                                                                     
MON_TT_CONSISTENCY_DAT_11                                                       
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_COMMENT_DAT                 SPLIT PARTITION                    
RUN_COMMENT_DAT_09                                                              
 AT (          0 ) INTO (PARTITION                                              
RUN_COMMENT_DAT_10                                                              
, PARTITION                                                                     
RUN_COMMENT_DAT_11                                                              
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                
ALTER TABLE  RUN_CONFIG_DAT                  SPLIT PARTITION                    
RUN_CONFIG_DAT_0                                                               
 AT (      50960 ) INTO (PARTITION                                              
RUN_CONFIG_DAT_10                                                               
, PARTITION                                                                     
RUN_CONFIG_DAT_11                                                               
 TABLESPACE CMS_ECAL_COND_2011_DATA) UPDATE GLOBAL INDEXES;                     
                                                                                

97 rows selected.

