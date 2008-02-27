$(header)

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
LIBRARY work;
USE work.muon_types.ALL;

PACKAGE $(particle)_setup IS
$(eta_4)
$(phi_l_4)
$(phi_h_4)
$(eta_3)
$(phi_l_3)
$(phi_h_3)
$(eta_2_s)
$(phi_l_2_s)
$(phi_h_2_s)
$(eta_2_wsc)
$(phi_l_2_wsc)
$(phi_h_2_wsc)
$(delta_eta)
$(delta_phi)
$(eta_1_s)
$(phi_l_1_s)
$(phi_h_1_s)
END $(particle)_setup;
