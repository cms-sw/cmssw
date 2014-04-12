$(header)

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;

USE work.types_pkg.ALL;
USE work.muon_types.ALL;

PACKAGE def_val_pkg IS

-- default-values for 8-bit registers
-- all types based on STD_LOGIC_VECTOR(7 DOWNTO 0),
-- converted to string in reg_rw_def_val.vhd because
-- of use in lpm_ff as string

-- ********************************************************
-- default-value for calo register (IEG)

$(muon_def_vals)

$(calo_def_vals)

$(esums_def_vals)

$(jets_def_vals)

END def_val_pkg;
