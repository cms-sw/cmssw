--                                                                             
-- File: CMS_ECAL_HV_PVSS_COND.sql                                             
--                                                                             
                                                                               
--                                                                             
-- DCSLASTVALUE_VOLTAGE                                                        
--                                                                             
                                                                               
CREATE TABLE DCSLASTVALUE_VOLTAGE AS                                           
SELECT DISTINCT                                                                
       LOGIC_ID,                                                               
       VMON,                                                                   
       SINCE                                                                   
FROM   PVSS_HV_VMON_DAT                                                        
WHERE  VMON IS NOT NULL                                                        
AND    (LOGIC_ID, SINCE) IN                                                    
       (                                                                       
        SELECT LOGIC_ID,                                                       
               MAX(SINCE)                                                      
        FROM   PVSS_HV_VMON_DAT                                                
        WHERE  VMON IS NOT NULL                                                
        GROUP  BY LOGIC_ID                                                     
       )                                                                       
ORDER  BY LOGIC_ID                                                             
/                                                                              
                                                                               
GRANT SELECT ON DCSLASTVALUE_VOLTAGE TO PUBLIC                                 
/                                                                              
                                                                               
CREATE OR REPLACE TRIGGER PVSS_HV_VMON_DAT_INSERT                              
  BEFORE INSERT ON PVSS_HV_VMON_DAT                                            
  FOR EACH ROW                                                                 
WHEN (NEW.VMON IS NOT NULL)                                                    
DECLARE                                                                        
  CHANGED_DP PVSS_HV_VMON_DAT.LOGIC_ID%TYPE;                                   
BEGIN                                                                          
  UPDATE DCSLASTVALUE_VOLTAGE SET                                              
    VMON = :NEW.VMON,                                                          
    SINCE = :NEW.SINCE                                                         
  WHERE LOGIC_ID = :NEW.LOGIC_ID                                               
  RETURNING LOGIC_ID INTO CHANGED_DP;                                          
  IF CHANGED_DP IS NULL THEN                                                   
    INSERT INTO DCSLASTVALUE_VOLTAGE VALUES                                    
    (                                                                          
     :NEW.LOGIC_ID,                                                            
     :NEW.VMON,                                                                
     :NEW.SINCE                                                                
    );                                                                         
  END IF;                                                                      
END PVSS_HV_VMON_DAT_INSERT;                                                   
/                                                                              

