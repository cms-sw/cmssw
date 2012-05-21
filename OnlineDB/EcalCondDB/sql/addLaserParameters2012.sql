CREATE TABLE ecal_laser_configuration_cp AS
	SELECT * FROM ecal_laser_configuration
/

ALTER TABLE ecal_laser_configuration ADD (
	IR_LASER_POSTTRIG NUMBER DEFAULT NULL,
        BLUE2_LASER_POSTTRIG NUMBER DEFAULT NULL,
	GREEN_LASER_POSTTRIG NUMBER DEFAULT NULL,
	IR_LASER_PHASE NUMBER DEFAULT NULL,
	BLUE2_LASER_PHASE NUMBER DEFAULT NULL,
	GREEN_LASER_PHASE NUMBER DEFAULT NULL,
	WTE_2_LED_SOAK_DELAY NUMBER DEFAULT -2,
	LED_POSTSCALE NUMBER DEFAULT -2,
	WTE_2_IR_LASER NUMBER(4) DEFAULT NULL,
	WTE_2_BLUE2_LASER NUMBER(4) DEFAULT NULL,
	WTE_2_GREEN_LASER NUMBER(4) DEFAULT NULL
)
/

SPOOL UPDATE.sql
SET LINE 180
SET HEAD OFF
SET FEED OFF
select 'update ecal_laser_configuration set IR_LASER_POSTTRIG=' || POSTTRIG || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set BLUE2_LASER_POSTTRIG=' || POSTTRIG || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set GREEN_LASER_POSTTRIG=' || POSTTRIG || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set IR_LASER_PHASE=' || LASER_PHASE || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set BLUE2_LASER_PHASE=' || LASER_PHASE || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set GREEN_LASER_PHASE=' || LASER_PHASE || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set WTE_2_IR_LASER=' || WTE2_LASER_DELAY || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set WTE_2_BLUE2_LASER=' || WTE2_LASER_DELAY || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
select 'update ecal_laser_configuration set WTE_2_GREEN_LASER=' || WTE2_LASER_DELAY || ' where LASER_CONFIGURATION_ID=' || LASER_CONFIGURATION_ID || ';' from ecal_laser_configuration order by laser_configuration_id
/
SPOOL OFF

SET FEED ON
SET HEAD ON
@UPDATE
/

ALTER TABLE ecal_laser_configuration 
	RENAME COLUMN POSTTRIG TO BLUE_LASER_POSTTRIG
/
ALTER TABLE ecal_laser_configuration 
	RENAME COLUMN LASER_PHASE TO BLUE_LASER_PHASE
/
ALTER TABLE ecal_laser_configuration 
	RENAME COLUMN WTE2_LASER_DELAY TO WTE_2_BLUE_LASER
/
ALTER TABLE ecal_laser_configuration 
	RENAME COLUMN POWER_SETTING TO BLUE_LASER_POWER
/
ALTER TABLE ecal_laser_configuration 
	RENAME COLUMN RED_LASER_POWER TO BLUE2_LASER_POWER
/
ALTER TABLE ecal_laser_configuration 
	RENAME COLUMN RED_LASER_LOG_ATTENUATOR TO BLUE2_LASER_LOG_ATTENUATOR
/

