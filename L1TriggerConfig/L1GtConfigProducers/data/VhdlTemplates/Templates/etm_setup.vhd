$(header)

LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
LIBRARY work;
USE work.types_pkg.ALL;

PACKAGE $(particle)_setup IS

CONSTANT $(particle)_phi: $(particle)_phi_arr := (
$(phi)
OTHERS => X"00000000000000000000000000000000"
);

END $(particle)_setup;
