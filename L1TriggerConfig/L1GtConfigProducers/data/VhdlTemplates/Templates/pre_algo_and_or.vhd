$(header)

------------------------------------------------------------
--                                                        --
-- LOGIC CORE: GTL-9U-module condition/algo chip logic    --
-- MODULE NAME: algo_and_or                               --
-- INSTITUTION: Hephy Vienna                              --
-- DESIGNER: H. Bergauer                                  --
--                                                        --
-- VERSION: V4.0                                          --
-- DATE: 06 2004                                          --
--                                                        --
-- FUNCTIONAL DESCRIPTION:                                --
-- algo logic (and-or)                                    --
--                                                        --
------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY algo_and_or IS
	GENERIC	(
			nr_ieg_4 : integer := 1;
			nr_ieg_2_s : integer := 1;
			nr_ieg_2_wsc : integer := 1;
			nr_ieg_1_s : integer := 1;
			nr_eg_4 : integer := 1;
			nr_eg_2_s : integer := 1;
			nr_eg_2_wsc : integer := 1;
			nr_eg_1_s : integer := 1;
			nr_jet_4 : integer := 1;
			nr_jet_2_s : integer := 1;
			nr_jet_2_wsc : integer := 1;
			nr_jet_1_s : integer := 1;
			nr_tau_4 : integer := 1;
			nr_tau_2_s : integer := 1;
			nr_tau_2_wsc : integer := 1;
			nr_tau_1_s : integer := 1;
			nr_fwdjet_4 : integer := 1;
			nr_fwdjet_2_s : integer := 1;
			nr_fwdjet_2_wsc : integer := 1;
			nr_fwdjet_1_s : integer := 1;
			nr_muon_4 : integer := 1;
			nr_muon_2_s : integer := 1;
			nr_muon_2_wsc : integer := 1;
			nr_muon_1_s : integer := 1;
			nr_muon_3 : integer := 1;
			nr_jet_cnts_0_cond : integer := 1;
			nr_jet_cnts_1_cond : integer := 1;
			nr_jet_cnts_2_cond : integer := 1;
			nr_jet_cnts_3_cond : integer := 1;
			nr_jet_cnts_4_cond : integer := 1;
			nr_jet_cnts_5_cond : integer := 1;
			nr_jet_cnts_6_cond : integer := 1;
			nr_jet_cnts_7_cond : integer := 1;
			nr_jet_cnts_8_cond : integer := 1;
			nr_jet_cnts_9_cond : integer := 1;
			nr_jet_cnts_10_cond : integer := 1;
			nr_jet_cnts_11_cond : integer := 1;
			nr_ett_cond : integer := 1;
			nr_etm_cond : integer := 1;
			nr_htt_cond : integer := 1);
	PORT(
		ieg_4		: IN	STD_LOGIC_VECTOR(nr_ieg_4 DOWNTO 0);
		ieg_2_s		: IN	STD_LOGIC_VECTOR(nr_ieg_2_s DOWNTO 0);
		ieg_2_wsc	: IN	STD_LOGIC_VECTOR(nr_ieg_2_wsc DOWNTO 0);
		ieg_1_s		: IN	STD_LOGIC_VECTOR(nr_ieg_1_s DOWNTO 0);
		eg_4		: IN	STD_LOGIC_VECTOR(nr_eg_4 DOWNTO 0);
		eg_2_s		: IN	STD_LOGIC_VECTOR(nr_eg_2_s DOWNTO 0);
		eg_2_wsc	: IN	STD_LOGIC_VECTOR(nr_eg_2_wsc DOWNTO 0);
		eg_1_s		: IN	STD_LOGIC_VECTOR(nr_eg_1_s DOWNTO 0);
		jet_4		: IN	STD_LOGIC_VECTOR(nr_jet_4 DOWNTO 0);
		jet_2_s		: IN	STD_LOGIC_VECTOR(nr_jet_2_s DOWNTO 0);
		jet_2_wsc	: IN	STD_LOGIC_VECTOR(nr_jet_2_wsc DOWNTO 0);
		jet_1_s		: IN	STD_LOGIC_VECTOR(nr_jet_1_s DOWNTO 0);
		tau_4		: IN	STD_LOGIC_VECTOR(nr_tau_4 DOWNTO 0);
		tau_2_s		: IN	STD_LOGIC_VECTOR(nr_tau_2_s DOWNTO 0);
		tau_2_wsc	: IN	STD_LOGIC_VECTOR(nr_tau_2_wsc DOWNTO 0);
		tau_1_s		: IN	STD_LOGIC_VECTOR(nr_tau_1_s DOWNTO 0);
		fwdjet_4	: IN	STD_LOGIC_VECTOR(nr_fwdjet_4 DOWNTO 0);
		fwdjet_2_s	: IN	STD_LOGIC_VECTOR(nr_fwdjet_2_s DOWNTO 0);
		fwdjet_2_wsc: IN	STD_LOGIC_VECTOR(nr_fwdjet_2_wsc DOWNTO 0);
		fwdjet_1_s	: IN	STD_LOGIC_VECTOR(nr_fwdjet_1_s DOWNTO 0);
		muon_4		: IN	STD_LOGIC_VECTOR(nr_muon_4 DOWNTO 0);
		muon_2_s	: IN	STD_LOGIC_VECTOR(nr_muon_2_s DOWNTO 0);
		muon_2_wsc	: IN	STD_LOGIC_VECTOR(nr_muon_2_wsc DOWNTO 0);
		muon_1_s	: IN	STD_LOGIC_VECTOR(nr_muon_1_s DOWNTO 0);
		muon_3		: IN	STD_LOGIC_VECTOR(nr_muon_3 DOWNTO 0);
		jet_cnts_0_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_0_cond DOWNTO 0);
		jet_cnts_1_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_1_cond DOWNTO 0);
		jet_cnts_2_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_2_cond DOWNTO 0);
		jet_cnts_3_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_3_cond DOWNTO 0);
		jet_cnts_4_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_4_cond DOWNTO 0);
		jet_cnts_5_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_5_cond DOWNTO 0);
		jet_cnts_6_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_6_cond DOWNTO 0);
		jet_cnts_7_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_7_cond DOWNTO 0);
		jet_cnts_8_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_8_cond DOWNTO 0);
		jet_cnts_9_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_9_cond DOWNTO 0);
		jet_cnts_10_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_10_cond DOWNTO 0);
		jet_cnts_11_cond:  IN	STD_LOGIC_VECTOR(nr_jet_cnts_11_cond DOWNTO 0);
		ett_cond:  IN	STD_LOGIC_VECTOR(nr_ett_cond DOWNTO 0);
		etm_cond:  IN	STD_LOGIC_VECTOR(nr_etm_cond DOWNTO 0);
		htt_cond:  IN	STD_LOGIC_VECTOR(nr_htt_cond DOWNTO 0);
		algo_s_reg	: OUT	STD_LOGIC_VECTOR(2 DOWNTO 0);
		algo_reg	: INOUT	STD_LOGIC_VECTOR(95 DOWNTO 0));
END algo_and_or;
ARCHITECTURE rtl OF algo_and_or IS
	SIGNAL pre_algo_a : STD_LOGIC_VECTOR(96 DOWNTO 1);
BEGIN

-- "NO ALGO"-bit
ALGO_S_REG(0) <= NOT(
  			ALGO_REG(0) OR ALGO_REG(1) OR ALGO_REG(2) OR ALGO_REG(3)
  			OR
  			ALGO_REG(4) OR ALGO_REG(5) OR ALGO_REG(6) OR ALGO_REG(7)
  			OR
  			ALGO_REG(8) OR ALGO_REG(9) OR ALGO_REG(10) OR ALGO_REG(11)
  			OR
  			ALGO_REG(12) OR ALGO_REG(13) OR ALGO_REG(14) OR ALGO_REG(15)
  			OR
  			ALGO_REG(16) OR ALGO_REG(17) OR ALGO_REG(18) OR ALGO_REG(19)
  			OR
  			ALGO_REG(20) OR ALGO_REG(21) OR ALGO_REG(22) OR ALGO_REG(23)
  			OR
  			ALGO_REG(24) OR ALGO_REG(25) OR ALGO_REG(26) OR ALGO_REG(27)
  			OR
  			ALGO_REG(28) OR ALGO_REG(29) OR ALGO_REG(30) OR ALGO_REG(31));

ALGO_S_REG(1) <= NOT(
  			ALGO_REG(32) OR ALGO_REG(33) OR ALGO_REG(34) OR ALGO_REG(53)
  			OR
  			ALGO_REG(36) OR ALGO_REG(37) OR ALGO_REG(38) OR ALGO_REG(39)
  			OR
  			ALGO_REG(40) OR ALGO_REG(41) OR ALGO_REG(42) OR ALGO_REG(43)
  			OR
  			ALGO_REG(44) OR ALGO_REG(45) OR ALGO_REG(46) OR ALGO_REG(47)
  			OR
  			ALGO_REG(48) OR ALGO_REG(49) OR ALGO_REG(50) OR ALGO_REG(51)
  			OR
  			ALGO_REG(52) OR ALGO_REG(53) OR ALGO_REG(54) OR ALGO_REG(55)
  			OR
  			ALGO_REG(56) OR ALGO_REG(57) OR ALGO_REG(58) OR ALGO_REG(59)
  			OR
  			ALGO_REG(60) OR ALGO_REG(61) OR ALGO_REG(62) OR ALGO_REG(63));

ALGO_S_REG(2) <= NOT(
  			ALGO_REG(64) OR ALGO_REG(65) OR ALGO_REG(66) OR ALGO_REG(67)
  			OR
  			ALGO_REG(68) OR ALGO_REG(69) OR ALGO_REG(70) OR ALGO_REG(71)
  			OR
  			ALGO_REG(72) OR ALGO_REG(73) OR ALGO_REG(74) OR ALGO_REG(75)
  			OR
  			ALGO_REG(76) OR ALGO_REG(77) OR ALGO_REG(78) OR ALGO_REG(79)
  			OR
  			ALGO_REG(80) OR ALGO_REG(81) OR ALGO_REG(82) OR ALGO_REG(83)
  			OR
  			ALGO_REG(84) OR ALGO_REG(85) OR ALGO_REG(86) OR ALGO_REG(87)
  			OR
  			ALGO_REG(88) OR ALGO_REG(89) OR ALGO_REG(90) OR ALGO_REG(91)
  			OR
  			ALGO_REG(92) OR ALGO_REG(93) OR ALGO_REG(94) OR ALGO_REG(95));

-- ***************************************************************

ALGO_REG(95 DOWNTO 0) <= pre_algo_a(96 DOWNTO 1);

$(prealgos)

END ARCHITECTURE rtl;
