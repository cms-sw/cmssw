DROP DATABASE LINK lhclogdb;

CREATE DATABASE LINK lhclogdb CONNECT TO measdb_pub IDENTIFIED BY meas2005 USING 'sunlhclog01.cern.ch:1521/LHCLOGDB';

SELECT count(*) FROM v_client_spsea_vars_last_vals@lhclogdb;

CREATE VIEW beam_source AS
SELECT * FROM v_client_spsea_vars_last_vals@lhclogdb WHERE beamline IN ('H4', 'H2');

SELECT count(*) FROM beam_source;
