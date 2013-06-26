/*
 *  Creates beam file to energy correspondence
 */

CREATE TABLE beamfile_to_energy_def (
  def_id		NUMBER(10),
  beam_file             varchar(100), 
  energy        	NUMBER(10),
  particle              varchar(20),
  special_settings      varchar(100)  	
);

ALTER TABLE beamfile_to_energy_def ADD CONSTRAINT beamfile_to_energy_def_pk PRIMARY KEY (def_id);
CREATE SEQUENCE beamfile_to_energy_sq INCREMENT BY 1 START WITH 1;



INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.001', 150.0 , 'p+ pi+', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.002', 350.0 , 'p+ pi+', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.005', 50.0 , 'p+ pi+', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.021', 100.0 , 'electrons', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.022', 200.0 , 'electrons', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.048', 150.0 , 'muons +', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.100', 20.0 , 'electrons', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.001', 150.0 , 'P+PI+H', 'FM,focus C8(OMR) collimators closed -> poor man muon beam');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.002', 120.0 , 'electrons', 'FM(@-4.52mr)');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.003', 20.0 , 'electrons', 'FM(@-4.52mr) P=20.000');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.004', 30.0 , 'electrons', 'FM(@-4.52mr) P=29.998');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.005', 50.0 , 'electrons', 'FM(@-4.52mr) P=49.988');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.006', 70.0 , 'electrons', 'FM(@-4.52mr) P=69.955');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.007', 150.0 , 'electrons', 'FM(@-4.52mr) not sure!');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.008', 80.0 , 'electrons', 'FM(@-4.52mr) P=79.766');
 INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.009', 0.0 , '', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4A.900', 20.0 , 'positrons', '');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.011', 90.0 , 'electrons', 'FM(@-4.52mr),P=89.835');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.012', 120.0 , 'electrons', 'FM(@-4.52mr),P=119.610');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.013', 150.0 , 'electrons', 'FM(@-4.52mr),P=149.053');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.014', 170.0 , 'electrons', 'FM(@-4.52mr),P=168.446');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.016', 200.0 , 'electrons', 'FM(@-4.52mr),P=197.055');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.017', 230.0 , 'electrons', 'FM(@-4.52mr),P=224.923');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.018', 250.0 , 'electrons', 'FM(@-4.52mr),P=242.996');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.020', 150.0 , 'P+PI+H', 'FM,focus C8(OMR), linearity run');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.021', 10.0 , 'electrons', 'FM(@-4.52mr),linearity run P=10.000');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.022', 20.0 , 'electrons', 'FM(@-4.52mr),linearity run P=20.000');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.023', 30.0 , 'electrons', 'FM(@-4.52mr),linearity run P=29.998');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.024', 50.0 , 'electrons', 'FM(@-4.52mr),linearity run P=49.981');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.025', 70.0 , 'electrons', 'FM(@-4.52mr),linearity run P=69.926');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.026', 90.0 , 'electrons', 'FM(@-4.52mr),linearity run P=89.799');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.027', 120.0 , 'electrons', 'FM(@-4.52mr),linearity run P=119.369');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.028', 150.0 , 'electrons', 'FM(@-4.52mr),linearity run P=148.474');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.029', 170.0 , 'electrons', 'FM(@-4.52mr),linearity run P=167.503');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.030', 15.0 , 'electrons', 'FM(@-4.52mr),linearity run P=15.00');

INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.400', 400.0 , '', 'magnet test');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.401', 400.0 , '', 'magnet polarity test');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.902', 120.0 , 'electrons', 'FM(@-4.52mr),P=119.610');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.903', 20.0 , 'electrons', 'FM(@-4.52mr),P=20.000');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.904', 30.0 , 'electrons', 'FM(@-4.52mr),P=29.998');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.905', 50.0 , 'electrons', 'FM(@-4.52mr),P=49.988');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.906', 70.0 , 'electrons', 'FM(@-4.52mr),P=69.955');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.908', 80.0 , 'electrons', 'FM(@-4.52mr),P=79.923');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.911', 90.0 , 'electrons', 'FM(@-4.52mr),P=89.876');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.912', 120.0 , 'electrons', 'FM(@-4.52mr),P=119.610');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.913', 150.0 , 'electrons', 'FM(@-4.52mr),P=149.053');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.914', 170.0 , 'electrons', 'FM(@-4.52mr),P=168.446');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.916', 200.0 , 'electrons', 'FM(@-4.52mr),P=197.055');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.917', 230.0 , 'electrons', 'FM(@-4.52mr),P=224.923');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.918', 250.0 , 'electrons', 'FM(@-4.52mr),P=242.996');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.A', 450.0 , '', 'P4 TO T24(2.51mr)-NOMINAL KOL');
INSERT INTO beamfile_to_energy_def(DEF_ID, beam_file, energy, particle, special_settings ) values (beamfile_to_energy_SQ.NextVal, 'H4C.RTEST', 200.0 ,'',  'P4 TO T24(2.51mr)-NOMINAL KOL');


