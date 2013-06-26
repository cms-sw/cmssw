$(header)

------------------------------------------------------------
--                                                        --
-- LOGIC CORE: GTL-9U-module condition/algo chip logic    --
-- MODULE NAME: cond_chip                                 --
-- INSTITUTION: Hephy Vienna                              --
-- DESIGNER: H. Bergauer                                  --
--                                                        --
-- VHDL-LIBRARY VERSION: V11.x                            --
-- DATE: 02 2008                                          --
--                                                        --
-- FUNCTIONAL DESCRIPTION:                                --
-- top of hierarchy for condition-chip logic of GTL-9U    --
--                                                        --
-- REVISION HISTORY:                                      --
-- Version: HB160306                                      --
-- |--> algo-memory as dual port memory with              --
-- |    6 x 1024 x 16 bits (16 bits vdata)                --
-- |--> PLL for 80 MHz clock for calo/muon input data     --
-- |--> DTACK for calos and muons                         --
-- |--> no BERR generated                                 --
-- Version: HB301107                                      --
-- |--> rw-register with parameters for read, dtack,      --
--      and default-values (with reset-option)            --
-- Version: HB150208                                      --
-- |--> INCLOCK_PERIOD implemented                        --
--                                                        --
------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
LIBRARY altera;
USE altera.maxplus2.ALL;

-- for "altclklock" - HB111105
LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

-- for 80MHz input-register - HB111105
LIBRARY lpm;
USE lpm.lpm_components.ALL;

USE work.cond_chip_pkg.ALL; 
USE work.def_val_pkg.ALL; 
USE work.algo_components.ALL; 
USE work.calo_condition_pkg.ALL;
USE work.muon_condition_pkg.ALL;
USE work.jc_esums_pkg.ALL;
USE work.basics_pkg.ALL;
USE work.vme_pkg.ALL;

ENTITY cond_chip IS
	PORT(
		CLK40	: IN	STD_LOGIC;
		CLK80	: IN	STD_LOGIC;
		CA113		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA124		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA213		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA224		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA313		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA324		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA413		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA424		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA513		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA524		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA613		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA624		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA713		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA724		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA813		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA824		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA913		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA924		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA1013		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		CA1024		: IN	STD_LOGIC_VECTOR(15 DOWNTO 0);
		MU1			: IN	STD_LOGIC_VECTOR(25 DOWNTO 0);
		MU3			: IN	STD_LOGIC_VECTOR(25 DOWNTO 0);
		ENCOND		: IN	STD_LOGIC;
		ENALGO		: IN	STD_LOGIC;
		WRCOND		: IN	STD_LOGIC;
		BCRES		: IN	STD_LOGIC;
		L1A			: IN	STD_LOGIC;
		L1RESET		: IN	STD_LOGIC;
		CLKLKEN		: IN	STD_LOGIC;
		ADDR		: IN	STD_LOGIC_VECTOR(21 DOWNTO 1);
		ADDR_0_SIM	: IN	STD_LOGIC; -- for quartus simulator
		VDATA		: INOUT	STD_LOGIC_VECTOR(15 DOWNTO 0);
		RESERVEVME	: OUT	STD_LOGIC;
		RESERVE1	: OUT	STD_LOGIC_VECTOR(15 DOWNTO 0);
		RESERVE2	: OUT	STD_LOGIC_VECTOR(15 DOWNTO 0);
		RESERVE3	: OUT	STD_LOGIC_VECTOR(15 DOWNTO 0);
		TEST0		: OUT	STD_LOGIC;
		TEST1		: OUT	STD_LOGIC;
		TEST2		: OUT	STD_LOGIC;
		TEST3		: OUT	STD_LOGIC;
		CLKLOCKED	: OUT	STD_LOGIC;
		STAT		: OUT	STD_LOGIC_VECTOR(1 DOWNTO 0);
		NDTACK		: OUT	STD_LOGIC;
		ALGOSTROB	: OUT	STD_LOGIC_VECTOR(2 DOWNTO 0);
		ALGO		: OUT	STD_LOGIC_VECTOR(95 DOWNTO 0));
END cond_chip;
ARCHITECTURE rtl OF cond_chip IS
-- *********************************************************
-- NEW signals for rw-registers with dtack
	CONSTANT dtack_inst : boolean := true;
	SIGNAL reset: STD_LOGIC := '0';
	SIGNAL ENCOND_int: STD_LOGIC;
-- dtack-signals for ieg
	SIGNAL dtack_ieg_1_s : STD_LOGIC_VECTOR(nr_ieg_1_s DOWNTO 0);
	SIGNAL dtack_ieg_2_s : STD_LOGIC_VECTOR(nr_ieg_2_s DOWNTO 0);
	SIGNAL dtack_ieg_2_wsc : STD_LOGIC_VECTOR(nr_ieg_2_wsc DOWNTO 0);
	SIGNAL dtack_ieg_4 : STD_LOGIC_VECTOR(nr_ieg_4 DOWNTO 0);
-- dtack-signals for eg
	SIGNAL dtack_eg_1_s : STD_LOGIC_VECTOR(nr_eg_1_s DOWNTO 0);
	SIGNAL dtack_eg_2_s : STD_LOGIC_VECTOR(nr_eg_2_s DOWNTO 0);
	SIGNAL dtack_eg_2_wsc : STD_LOGIC_VECTOR(nr_eg_2_wsc DOWNTO 0);
	SIGNAL dtack_eg_4 : STD_LOGIC_VECTOR(nr_eg_4 DOWNTO 0);
-- dtack-signals for jet
	SIGNAL dtack_jet_1_s : STD_LOGIC_VECTOR(nr_jet_1_s DOWNTO 0);
	SIGNAL dtack_jet_2_s : STD_LOGIC_VECTOR(nr_jet_2_s DOWNTO 0);
	SIGNAL dtack_jet_2_wsc : STD_LOGIC_VECTOR(nr_jet_2_wsc DOWNTO 0);
	SIGNAL dtack_jet_4 : STD_LOGIC_VECTOR(nr_jet_4 DOWNTO 0);
-- dtack-signals for fwdjet
	SIGNAL dtack_fwdjet_1_s : STD_LOGIC_VECTOR(nr_fwdjet_1_s DOWNTO 0);
	SIGNAL dtack_fwdjet_2_s : STD_LOGIC_VECTOR(nr_fwdjet_2_s DOWNTO 0);
	SIGNAL dtack_fwdjet_2_wsc : STD_LOGIC_VECTOR(nr_fwdjet_2_wsc DOWNTO 0);
	SIGNAL dtack_fwdjet_4 : STD_LOGIC_VECTOR(nr_fwdjet_4 DOWNTO 0);
-- dtack-signals for tau
	SIGNAL dtack_tau_1_s : STD_LOGIC_VECTOR(nr_tau_1_s DOWNTO 0);
	SIGNAL dtack_tau_2_s : STD_LOGIC_VECTOR(nr_tau_2_s DOWNTO 0);
	SIGNAL dtack_tau_2_wsc : STD_LOGIC_VECTOR(nr_tau_2_wsc DOWNTO 0);
	SIGNAL dtack_tau_4 : STD_LOGIC_VECTOR(nr_tau_4 DOWNTO 0);
-- dtack-signals for muons
	SIGNAL dtack_muon_1_s : STD_LOGIC_VECTOR(nr_muon_1_s DOWNTO 0);
	SIGNAL dtack_muon_2_s : STD_LOGIC_VECTOR(nr_muon_2_s DOWNTO 0);
	SIGNAL dtack_muon_2_wsc : STD_LOGIC_VECTOR(nr_muon_2_wsc DOWNTO 0);
	SIGNAL dtack_muon_3 : STD_LOGIC_VECTOR(nr_muon_3 DOWNTO 0);
	SIGNAL dtack_muon_4 : STD_LOGIC_VECTOR(nr_muon_4 DOWNTO 0);
-- dtack-signals for esums
	SIGNAL dtack_jet_cnts_0 : STD_LOGIC_VECTOR(nr_jet_cnts_0_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_1 : STD_LOGIC_VECTOR(nr_jet_cnts_1_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_2 : STD_LOGIC_VECTOR(nr_jet_cnts_2_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_3 : STD_LOGIC_VECTOR(nr_jet_cnts_3_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_4 : STD_LOGIC_VECTOR(nr_jet_cnts_4_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_5 : STD_LOGIC_VECTOR(nr_jet_cnts_5_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_6 : STD_LOGIC_VECTOR(nr_jet_cnts_6_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_7 : STD_LOGIC_VECTOR(nr_jet_cnts_7_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_8 : STD_LOGIC_VECTOR(nr_jet_cnts_8_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_9 : STD_LOGIC_VECTOR(nr_jet_cnts_9_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_10 : STD_LOGIC_VECTOR(nr_jet_cnts_10_cond DOWNTO 0);
	SIGNAL dtack_jet_cnts_11 : STD_LOGIC_VECTOR(nr_jet_cnts_11_cond DOWNTO 0);
-- dtack-signals for esums
	SIGNAL dtack_ett : STD_LOGIC_VECTOR(nr_ett_cond DOWNTO 0);
	SIGNAL dtack_etm : STD_LOGIC_VECTOR(nr_etm_cond DOWNTO 0);
	SIGNAL dtack_htt : STD_LOGIC_VECTOR(nr_htt_cond DOWNTO 0);
-- *********************************************************
-- address signals
	SIGNAL addr_cond: STD_LOGIC_VECTOR(7 DOWNTO 0);
	SIGNAL addr_reg_name: STD_LOGIC_VECTOR(4 DOWNTO 0);
-- 80MHz input register signals
	SIGNAL ca113_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca124_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca213_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca224_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca313_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca324_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca413_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca424_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca513_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca524_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca613_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca624_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca713_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca724_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca813_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca824_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca913_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca924_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca1013_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca1024_ioc: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL mu1_ioc: STD_LOGIC_VECTOR(25 DOWNTO 0);
	SIGNAL mu3_ioc: STD_LOGIC_VECTOR(25 DOWNTO 0);
-- input_calos/input_muons register signals
	SIGNAL ca11_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca12_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca13_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca14_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca21_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca22_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca23_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca24_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca31_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca32_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca33_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca34_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca41_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca42_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca43_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca44_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca51_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca52_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca53_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca54_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca61_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca62_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca63_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca64_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca71_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca72_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca73_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca74_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca81_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca82_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca83_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca84_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca91_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca92_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca93_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca94_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca101_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca102_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca103_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL ca104_reg: STD_LOGIC_VECTOR(15 DOWNTO 0);
	SIGNAL mu1_reg: STD_LOGIC_VECTOR(25 DOWNTO 0);
	SIGNAL mu2_reg: STD_LOGIC_VECTOR(25 DOWNTO 0);
	SIGNAL mu3_reg: STD_LOGIC_VECTOR(25 DOWNTO 0);
	SIGNAL mu4_reg: STD_LOGIC_VECTOR(25 DOWNTO 0);
-- particle enable signals
	SIGNAL ieg_en : STD_LOGIC;
	SIGNAL eg_en : STD_LOGIC;
	SIGNAL jet_en : STD_LOGIC;
	SIGNAL tau_en : STD_LOGIC;
	SIGNAL fwdjet_en : STD_LOGIC;
	SIGNAL muon_en : STD_LOGIC;
-- IEG condition enable signals
	SIGNAL en_ieg_4: STD_LOGIC;
	SIGNAL en_ieg_2_s: STD_LOGIC;
	SIGNAL en_ieg_2_wsc: STD_LOGIC;
	SIGNAL en_ieg_1_s: STD_LOGIC;
-- EG condition enable signals
	SIGNAL en_eg_4: STD_LOGIC;
	SIGNAL en_eg_2_s: STD_LOGIC;
	SIGNAL en_eg_2_wsc: STD_LOGIC;
	SIGNAL en_eg_1_s: STD_LOGIC;
-- JET condition enable signals	
	SIGNAL en_jet_4: STD_LOGIC;
	SIGNAL en_jet_2_s: STD_LOGIC;
	SIGNAL en_jet_2_wsc: STD_LOGIC;
	SIGNAL en_jet_1_s: STD_LOGIC;
-- TAU condition enable signals	
	SIGNAL en_tau_4: STD_LOGIC;
	SIGNAL en_tau_2_s: STD_LOGIC;
	SIGNAL en_tau_2_wsc: STD_LOGIC;
	SIGNAL en_tau_1_s: STD_LOGIC;
-- FWDJET condition enable signals	
	SIGNAL en_fwdjet_4: STD_LOGIC;
	SIGNAL en_fwdjet_2_s: STD_LOGIC;
	SIGNAL en_fwdjet_2_wsc: STD_LOGIC;
	SIGNAL en_fwdjet_1_s: STD_LOGIC;
-- jet counts condition enable signals	
	SIGNAL en_jet_cnts_0, en_jet_cnts_1, en_jet_cnts_2: STD_LOGIC;
	SIGNAL en_jet_cnts_3, en_jet_cnts_4, en_jet_cnts_5: STD_LOGIC;
	SIGNAL en_jet_cnts_6, en_jet_cnts_7, en_jet_cnts_8: STD_LOGIC;
	SIGNAL en_jet_cnts_9, en_jet_cnts_10, en_jet_cnts_11: STD_LOGIC;
-- e_sums condition enable signals	
	SIGNAL en_ett_cond: STD_LOGIC;
	SIGNAL en_etm_cond: STD_LOGIC;
	SIGNAL en_htt_cond: STD_LOGIC;
-- MUON condition enable signals	
	SIGNAL en_muon_4: STD_LOGIC;
	SIGNAL en_muon_2_s: STD_LOGIC;
	SIGNAL en_muon_2_wsc: STD_LOGIC;
	SIGNAL en_muon_1_s: STD_LOGIC;
	SIGNAL en_muon_3: STD_LOGIC;
-- outputs of inputregisters IEG
	SIGNAL reg_ieg_et_1: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_ieg_et_2: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_ieg_et_3: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_ieg_et_4: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_ieg_eta_1: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_ieg_eta_2: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_ieg_eta_3: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_ieg_eta_4: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_ieg_phi_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_ieg_phi_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_ieg_phi_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_ieg_phi_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_ieg_sync_1: STD_LOGIC;
	SIGNAL reg_ieg_sync_2: STD_LOGIC;
	SIGNAL reg_ieg_sync_3: STD_LOGIC;
	SIGNAL reg_ieg_sync_4: STD_LOGIC;
-- outputs of inputregisters EG
	SIGNAL reg_eg_et_1: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_eg_et_2: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_eg_et_3: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_eg_et_4: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_eg_eta_1: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_eg_eta_2: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_eg_eta_3: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_eg_eta_4: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_eg_phi_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_eg_phi_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_eg_phi_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_eg_phi_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_eg_sync_1: STD_LOGIC;
	SIGNAL reg_eg_sync_2: STD_LOGIC;
	SIGNAL reg_eg_sync_3: STD_LOGIC;
	SIGNAL reg_eg_sync_4: STD_LOGIC;
-- outputs of inputregisters JET
	SIGNAL reg_jet_et_1: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_jet_et_2: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_jet_et_3: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_jet_et_4: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_jet_eta_1: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_jet_eta_2: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_jet_eta_3: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_jet_eta_4: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_jet_phi_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_phi_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_phi_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_phi_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_sync_1: STD_LOGIC;
	SIGNAL reg_jet_sync_2: STD_LOGIC;
	SIGNAL reg_jet_sync_3: STD_LOGIC;
	SIGNAL reg_jet_sync_4: STD_LOGIC;
-- outputs of inputregisters FWDJET
	SIGNAL reg_fwdjet_et_1: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_fwdjet_et_2: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_fwdjet_et_3: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_fwdjet_et_4: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_fwdjet_eta_1: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_fwdjet_eta_2: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_fwdjet_eta_3: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_fwdjet_eta_4: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_fwdjet_phi_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_fwdjet_phi_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_fwdjet_phi_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_fwdjet_phi_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_fwdjet_sync_1: STD_LOGIC;
	SIGNAL reg_fwdjet_sync_2: STD_LOGIC;
	SIGNAL reg_fwdjet_sync_3: STD_LOGIC;
	SIGNAL reg_fwdjet_sync_4: STD_LOGIC;
-- outputs of inputregisters TAU
	SIGNAL reg_tau_et_1: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_tau_et_2: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_tau_et_3: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_tau_et_4: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_tau_eta_1: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_tau_eta_2: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_tau_eta_3: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_tau_eta_4: STD_LOGIC_VECTOR(3 DOWNTO 0);
	SIGNAL reg_tau_phi_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_tau_phi_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_tau_phi_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_tau_phi_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_tau_sync_1: STD_LOGIC;
	SIGNAL reg_tau_sync_2: STD_LOGIC;
	SIGNAL reg_tau_sync_3: STD_LOGIC;
	SIGNAL reg_tau_sync_4: STD_LOGIC;
-- outputs of inputregisters "e_sums"
	SIGNAL reg_ett		: STD_LOGIC_VECTOR(12 DOWNTO 0);
	SIGNAL reg_htt		: STD_LOGIC_VECTOR(12 DOWNTO 0);
	SIGNAL reg_etm		: STD_LOGIC_VECTOR(12 DOWNTO 0);
	SIGNAL reg_etm_phi	: STD_LOGIC_VECTOR(6 DOWNTO 0);
	SIGNAL reg_esums_sync_1: STD_LOGIC;
	SIGNAL reg_esums_sync_2: STD_LOGIC;
	SIGNAL reg_esums_sync_3: STD_LOGIC;
	SIGNAL reg_esums_sync_4: STD_LOGIC;
-- outputs of inputregisters "jet-counters"
	SIGNAL reg_jet_cnts_0: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_5: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_6: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_7: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_8: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_9: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_10: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_11: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_jet_cnts_sync_1: STD_LOGIC;
	SIGNAL reg_jet_cnts_sync_2: STD_LOGIC;
	SIGNAL reg_jet_cnts_sync_3: STD_LOGIC;
	SIGNAL reg_jet_cnts_sync_4: STD_LOGIC;
-- outputs of inputregisters MUON
	SIGNAL reg_muon_pt_1: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_muon_pt_2: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_muon_pt_3: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_muon_pt_4: STD_LOGIC_VECTOR(4 DOWNTO 0);
	SIGNAL reg_muon_eta_1: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_muon_eta_2: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_muon_eta_3: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_muon_eta_4: STD_LOGIC_VECTOR(5 DOWNTO 0);
	SIGNAL reg_muon_phi_1: STD_LOGIC_VECTOR(7 DOWNTO 0);
	SIGNAL reg_muon_phi_2: STD_LOGIC_VECTOR(7 DOWNTO 0);
	SIGNAL reg_muon_phi_3: STD_LOGIC_VECTOR(7 DOWNTO 0);
	SIGNAL reg_muon_phi_4: STD_LOGIC_VECTOR(7 DOWNTO 0);
	SIGNAL reg_muon_qual_1: STD_LOGIC_VECTOR(2 DOWNTO 0);
	SIGNAL reg_muon_qual_2: STD_LOGIC_VECTOR(2 DOWNTO 0);
	SIGNAL reg_muon_qual_3: STD_LOGIC_VECTOR(2 DOWNTO 0);
	SIGNAL reg_muon_qual_4: STD_LOGIC_VECTOR(2 DOWNTO 0);
	SIGNAL reg_muon_iso_1: STD_LOGIC;
	SIGNAL reg_muon_iso_2: STD_LOGIC;
	SIGNAL reg_muon_iso_3: STD_LOGIC;
	SIGNAL reg_muon_iso_4: STD_LOGIC;
	SIGNAL reg_muon_mip_1: STD_LOGIC;
	SIGNAL reg_muon_mip_2: STD_LOGIC;
	SIGNAL reg_muon_mip_3: STD_LOGIC;
	SIGNAL reg_muon_mip_4: STD_LOGIC;
	SIGNAL reg_muon_sy_0_1: STD_LOGIC;
	SIGNAL reg_muon_sy_0_2: STD_LOGIC;
	SIGNAL reg_muon_sy_0_3: STD_LOGIC;
	SIGNAL reg_muon_sy_0_4: STD_LOGIC;
	SIGNAL reg_muon_sy_1_1: STD_LOGIC;
	SIGNAL reg_muon_sy_1_2: STD_LOGIC;
	SIGNAL reg_muon_sy_1_3: STD_LOGIC;
	SIGNAL reg_muon_sy_1_4: STD_LOGIC;
-- condition outputs
-- calos
	SIGNAL ieg_4: STD_LOGIC_VECTOR(nr_ieg_4 DOWNTO 0);
	SIGNAL ieg_2_s: STD_LOGIC_VECTOR(nr_ieg_2_s DOWNTO 0);
	SIGNAL ieg_2_wsc: STD_LOGIC_VECTOR(nr_ieg_2_wsc DOWNTO 0);
	SIGNAL ieg_1_s: STD_LOGIC_VECTOR(nr_ieg_1_s DOWNTO 0);
	SIGNAL eg_4: STD_LOGIC_VECTOR(nr_eg_4 DOWNTO 0);
	SIGNAL eg_2_s: STD_LOGIC_VECTOR(nr_eg_2_s DOWNTO 0);
	SIGNAL eg_2_wsc: STD_LOGIC_VECTOR(nr_eg_2_wsc DOWNTO 0);
	SIGNAL eg_1_s: STD_LOGIC_VECTOR(nr_eg_1_s DOWNTO 0);
	SIGNAL jet_4: STD_LOGIC_VECTOR(nr_jet_4 DOWNTO 0);
	SIGNAL jet_2_s: STD_LOGIC_VECTOR(nr_jet_2_s DOWNTO 0);
	SIGNAL jet_2_wsc: STD_LOGIC_VECTOR(nr_jet_2_wsc DOWNTO 0);
	SIGNAL jet_1_s: STD_LOGIC_VECTOR(nr_jet_1_s DOWNTO 0);
	SIGNAL tau_4: STD_LOGIC_VECTOR(nr_tau_4 DOWNTO 0);
	SIGNAL tau_2_s: STD_LOGIC_VECTOR(nr_tau_2_s DOWNTO 0);
	SIGNAL tau_2_wsc: STD_LOGIC_VECTOR(nr_tau_2_wsc DOWNTO 0);
	SIGNAL tau_1_s: STD_LOGIC_VECTOR(nr_tau_1_s DOWNTO 0);
	SIGNAL fwdjet_4: STD_LOGIC_VECTOR(nr_fwdjet_4 DOWNTO 0);
	SIGNAL fwdjet_2_s: STD_LOGIC_VECTOR(nr_fwdjet_2_s DOWNTO 0);
	SIGNAL fwdjet_2_wsc: STD_LOGIC_VECTOR(nr_fwdjet_2_wsc DOWNTO 0);
	SIGNAL fwdjet_1_s: STD_LOGIC_VECTOR(nr_fwdjet_1_s DOWNTO 0);
-- muons
	SIGNAL muon_4: STD_LOGIC_VECTOR(nr_muon_4 DOWNTO 0);
	SIGNAL muon_2_s: STD_LOGIC_VECTOR(nr_muon_2_s DOWNTO 0);
	SIGNAL muon_2_wsc: STD_LOGIC_VECTOR(nr_muon_2_wsc DOWNTO 0);
	SIGNAL muon_1_s: STD_LOGIC_VECTOR(nr_muon_1_s DOWNTO 0);
	SIGNAL muon_3: STD_LOGIC_VECTOR(nr_muon_3 DOWNTO 0);
-- jet-counters
	SIGNAL jet_cnts_0_cond: STD_LOGIC_VECTOR(nr_jet_cnts_0_cond DOWNTO 0);
	SIGNAL jet_cnts_1_cond: STD_LOGIC_VECTOR(nr_jet_cnts_1_cond DOWNTO 0);
	SIGNAL jet_cnts_2_cond: STD_LOGIC_VECTOR(nr_jet_cnts_2_cond DOWNTO 0);
	SIGNAL jet_cnts_3_cond: STD_LOGIC_VECTOR(nr_jet_cnts_3_cond DOWNTO 0);
	SIGNAL jet_cnts_4_cond: STD_LOGIC_VECTOR(nr_jet_cnts_4_cond DOWNTO 0);
	SIGNAL jet_cnts_5_cond: STD_LOGIC_VECTOR(nr_jet_cnts_5_cond DOWNTO 0);
	SIGNAL jet_cnts_6_cond: STD_LOGIC_VECTOR(nr_jet_cnts_6_cond DOWNTO 0);
	SIGNAL jet_cnts_7_cond: STD_LOGIC_VECTOR(nr_jet_cnts_7_cond DOWNTO 0);
	SIGNAL jet_cnts_8_cond: STD_LOGIC_VECTOR(nr_jet_cnts_8_cond DOWNTO 0);
	SIGNAL jet_cnts_9_cond: STD_LOGIC_VECTOR(nr_jet_cnts_9_cond DOWNTO 0);
	SIGNAL jet_cnts_10_cond: STD_LOGIC_VECTOR(nr_jet_cnts_10_cond DOWNTO 0);
	SIGNAL jet_cnts_11_cond: STD_LOGIC_VECTOR(nr_jet_cnts_11_cond DOWNTO 0);
-- e_sums
	SIGNAL ett_cond: STD_LOGIC_VECTOR(nr_ett_cond DOWNTO 0);
	SIGNAL etm_cond: STD_LOGIC_VECTOR(nr_etm_cond DOWNTO 0);
	SIGNAL htt_cond: STD_LOGIC_VECTOR(nr_htt_cond DOWNTO 0);
-- inputs of pre-algo outputregister
	SIGNAL algo_mux_in: STD_LOGIC_VECTOR(95 DOWNTO 0);
	SIGNAL algo_cond_mem: STD_LOGIC_VECTOR(95 DOWNTO 0);
	SIGNAL ALGO_OUT_REG_IN: STD_LOGIC_VECTOR(95 DOWNTO 0);
	SIGNAL ALGOSTROB_reg: STD_LOGIC_VECTOR(2 DOWNTO 0);
-- internal clock name
	SIGNAL clk_inp : STD_LOGIC;
	SIGNAL clk_cond : STD_LOGIC;
	SIGNAL clk_algo : STD_LOGIC;
-- DTACK/BERR signals
	SIGNAL dtack_calo_muon: STD_LOGIC;
	SIGNAL dtack_int: STD_LOGIC;
	SIGNAL ndtack_int: STD_LOGIC;
	SIGNAL DTACK_COND_MEM: STD_LOGIC;
	SIGNAL dtack_chip_id: STD_LOGIC;
	SIGNAL dtack_mux: STD_LOGIC;
-- muon charge logic signals
	SIGNAL sync_word : STD_LOGIC;
-- for m_cond_1_s
	SIGNAL pos_lut_1, pos_lut_2, pos_lut_3, pos_lut_4 : STD_LOGIC;
	SIGNAL neg_lut_1, neg_lut_2, neg_lut_3, neg_lut_4 : STD_LOGIC;
-- for m_cond_2_s and m_cond_2_wsc
	SIGNAL eq_lut_12, eq_lut_13, eq_lut_14 : STD_LOGIC;
	SIGNAL eq_lut_23, eq_lut_24, eq_lut_34 : STD_LOGIC;
	SIGNAL neq_lut_12, neq_lut_13, neq_lut_14 : STD_LOGIC;
	SIGNAL neq_lut_23, neq_lut_24, neq_lut_34 : STD_LOGIC;
-- for m_cond_3
	SIGNAL eq_lut_123, eq_lut_124, eq_lut_134, eq_lut_234 : STD_LOGIC;
	SIGNAL neq_lut_123, neq_lut_124, neq_lut_134, neq_lut_234 : STD_LOGIC;
-- for m_cond_4
	SIGNAL eq_lut_1234, pair_lut : STD_LOGIC;
-- internal pll names
	SIGNAL CLK40_PLL : STD_LOGIC;
	SIGNAL CLK80_PLL : STD_LOGIC;
	SIGNAL TEST80_PLL : STD_LOGIC;
	SIGNAL pll_locked : STD_LOGIC;
BEGIN
-- ************* DEFINITIONS *******************************
-- addresses
addr_cond(7 DOWNTO 2) <= addr(13 DOWNTO 8);
addr_cond(1 DOWNTO 0) <= addr(2 DOWNTO 1);
addr_reg_name(4 DOWNTO 0) <= addr(7 DOWNTO 3);
-- clocks
clk_inp <= CLK40_PLL;
clk_cond <= CLK40_PLL;
clk_algo <= CLK40_PLL;
-- Testoutputs
TEST0 <= dtack_jet_cnts_8(1-1);
--TEST0 <= CLK40;
TEST1 <= TEST80_PLL;
TEST2 <= pll_locked;
TEST3 <= CLK80;
-- status outputs, not used now !!!
STAT(0) <= '0';
STAT(1) <= '0';
-- PLL "locked" output
CLKLOCKED <= pll_locked;
-- reserve outputs (inputs) to VME- and REC-chips, not used now !!!
RESERVEVME <= '0';
RESERVE1 <= X"0000";
RESERVE2 <= X"0000";
RESERVE3 <= X"0000";
-- ************* PLL SECTION *******************************
-- HB 111105
-- altclklock einfacher, als altpll !!!
-- simulation schaut gut aus !!!
-- CLK40_PLL eingebaut, damit setup für calos_ioc und muons_ioc passt - HB010206 !!!
-- INCLOCK_PERIOD eingebaut für timing constraints - HB150208 !!!
pll_inst: altclklock
	GENERIC	MAP(INTENDED_DEVICE_FAMILY => "STRATIX",
		OPERATION_MODE => "NORMAL",
		INCLOCK_PERIOD => INCLOCK_PERIOD, -- see cond_chip_pkg.vhd !!!
		VALID_LOCK_MULTIPLIER => 1,
		CLOCK0_BOOST => 2, CLOCK1_BOOST => 2, CLOCK2_BOOST => 1)
	PORT MAP(
		inclock => CLK40, 
		clock0 => CLK80_PLL, 
		clock1 => TEST80_PLL,
		clock2 => CLK40_PLL, 
		locked => pll_locked);

-- ************* VME SECTION *******************************
-- REMARK: ENCOND for a write-cycle is made with DSPULS (25ns !!!) in VME-chip (V1004)
-- REMARK: ENCOND for a read-cycle is made with DSSYNC in VME-chip (V1004)
-- ENCOND_int is generated for proper use in rw-registers
call_encond_sync: bit_reg
	PORT MAP(clk_inp, ENCOND,
			ENCOND_int);
			
-- decoders for VME-registers
call_calo_dec: calo_decoder
	GENERIC MAP (rd_reg_inst)
	PORT MAP(
		ENCOND_int, WRCOND, addr(21 DOWNTO 14),
		ieg_en, eg_en, jet_en, tau_en, fwdjet_en,
		en_ieg_4, en_ieg_2_s, en_ieg_2_wsc, en_ieg_1_s,
		en_eg_4, en_eg_2_s, en_eg_2_wsc, en_eg_1_s,
		en_jet_4, en_jet_2_s, en_jet_2_wsc, en_jet_1_s,
		en_tau_4, en_tau_2_s, en_tau_2_wsc, en_tau_1_s,
		en_fwdjet_4, en_fwdjet_2_s, en_fwdjet_2_wsc,
		en_fwdjet_1_s);

call_muon_dec: muon_decoder
	GENERIC MAP (rd_reg_inst)
	PORT MAP(
		ENCOND_int, WRCOND, addr(21 DOWNTO 14),
		en_muon_4, en_muon_2_s, en_muon_2_wsc,
		en_muon_1_s, en_muon_3);

call_jc_es_dec: jc_es_decoder
	GENERIC MAP (rd_reg_inst)
	PORT MAP(
		ENCOND_int, WRCOND, addr(21 DOWNTO 14),
		en_jet_cnts_0, en_jet_cnts_1, en_jet_cnts_2,
		en_jet_cnts_3, en_jet_cnts_4, en_jet_cnts_5,
		en_jet_cnts_6, en_jet_cnts_7, en_jet_cnts_8,
		en_jet_cnts_9, en_jet_cnts_10, en_jet_cnts_11,
		en_ett_cond, en_etm_cond, en_htt_cond);

-- chip_id- and version-registers (read only)
call_chip_id_version: chip_id_version
	GENERIC	MAP(chip_id, version)
	PORT MAP(
		ENCOND_int, WRCOND, ADDR(21 DOWNTO 1), VDATA(7 DOWNTO 0),
		DTACK_chip_id);

-- ndtack_int logic from register
call_dtack_inst:
IF dtack_inst = true GENERATE
	call_dtack_calos_muon_or: dtack_calos_muon_or
		GENERIC MAP (
			nr_ieg_4, nr_ieg_2_s, nr_ieg_2_wsc, nr_ieg_1_s, 
			nr_eg_4, nr_eg_2_s, nr_eg_2_wsc, nr_eg_1_s, 
			nr_jet_4, nr_jet_2_s, nr_jet_2_wsc, nr_jet_1_s, 
			nr_tau_4, nr_tau_2_s, nr_tau_2_wsc, nr_tau_1_s, 
			nr_fwdjet_4, nr_fwdjet_2_s, nr_fwdjet_2_wsc, nr_fwdjet_1_s,
			nr_muon_4, nr_muon_2_s, nr_muon_2_wsc, nr_muon_1_s, nr_muon_3,
			nr_jet_cnts_0_cond, nr_jet_cnts_1_cond, nr_jet_cnts_2_cond,
			nr_jet_cnts_3_cond, nr_jet_cnts_4_cond, nr_jet_cnts_5_cond,
			nr_jet_cnts_6_cond, nr_jet_cnts_7_cond, nr_jet_cnts_8_cond,
			nr_jet_cnts_9_cond, nr_jet_cnts_10_cond, nr_jet_cnts_11_cond,
			nr_ett_cond, nr_etm_cond, nr_htt_cond)
		PORT MAP(
			dtack_ieg_1_s, dtack_ieg_2_s, dtack_ieg_2_wsc, dtack_ieg_4,
			dtack_eg_1_s, dtack_eg_2_s, dtack_eg_2_wsc, dtack_eg_4,
			dtack_jet_1_s, dtack_jet_2_s, dtack_jet_2_wsc, dtack_jet_4,
			dtack_tau_1_s, dtack_tau_2_s, dtack_tau_2_wsc, dtack_tau_4,
			dtack_fwdjet_1_s, dtack_fwdjet_2_s, dtack_fwdjet_2_wsc, dtack_fwdjet_4,
			dtack_muon_1_s, dtack_muon_2_s, dtack_muon_2_wsc, dtack_muon_3, dtack_muon_4,
			dtack_jet_cnts_0, dtack_jet_cnts_1, dtack_jet_cnts_2, dtack_jet_cnts_3, 
			dtack_jet_cnts_4, dtack_jet_cnts_5, dtack_jet_cnts_6, dtack_jet_cnts_7, 
			dtack_jet_cnts_8, dtack_jet_cnts_9, dtack_jet_cnts_10, dtack_jet_cnts_11, 
			dtack_ett, dtack_etm, dtack_htt,
			dtack_int);
END GENERATE call_dtack_inst;

NDTACK <= NOT (dtack_int OR dtack_chip_id OR DTACK_MUX OR DTACK_COND_MEM);

-- open drain outputs for NDTACK !!!
--ndtack_int <= NOT (dtack_int OR dtack_chip_id OR DTACK_MUX OR DTACK_COND_MEM);

--open_drain_ndtack: OPNDRN
--	PORT MAP(ndtack_int, NDTACK);

-- ********************************************************************************************************
-- BEGIN OF CONDITION-ALGO-LOGIC
-- ********************************************************************************************************
-- 
-- ************* INPUT DEFINITIONS *******************************
-- HB 111105
-- input-registers with CLK80_PLL implemented on calo- and muon-inputs
-- calorimeters
-- ca113 ... 80MHz data (channel 1, objects 1 and 3) from pin
-- ca113_ioc ... 80MHz data (channel 1, objects 1 and 3) output of register
calos_ioc_inst: calos_ioc
  		GENERIC MAP(16)
   		PORT MAP(CLK80_PLL,
   			ca113, ca124, ca213, ca224, ca313, ca324,  
   			ca413, ca424, ca513, ca524, ca613, ca624,  
   			ca713, ca724, ca813, ca824, ca913, ca924,  
   			ca1013, ca1024,  
			ca113_ioc, ca124_ioc, ca213_ioc, ca224_ioc,
			ca313_ioc, ca324_ioc, ca413_ioc, ca424_ioc,
			ca513_ioc, ca524_ioc, ca613_ioc, ca624_ioc,
			ca713_ioc, ca724_ioc, ca813_ioc, ca824_ioc,
			ca913_ioc, ca924_ioc, ca1013_ioc, ca1024_ioc);
muons_ioc_inst: muons_ioc
  		GENERIC MAP(26)
   		PORT MAP(CLK80_PLL,
   			mu1, mu3, 
			mu1_ioc, mu3_ioc);
-- ca41_reg ... 40MHz data, object 1 of channel 4
-- ca42_reg ... 40MHz data, object 2 of channel 4
ca113_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca113_ioc, ca11_reg, ca13_reg);
ca124_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca124_ioc, ca12_reg, ca14_reg);
ca213_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca213_ioc, ca21_reg, ca23_reg);
ca224_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca224_ioc, ca22_reg, ca24_reg);
ca313_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca313_ioc, ca31_reg, ca33_reg);
ca324_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca324_ioc, ca32_reg, ca34_reg);
ca413_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca413_ioc, ca41_reg, ca43_reg);
ca424_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca424_ioc, ca42_reg, ca44_reg);
ca513_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca513_ioc, ca51_reg, ca53_reg);
ca524_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca524_ioc, ca52_reg, ca54_reg);
ca613_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca613_ioc, ca61_reg, ca63_reg);
ca624_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca624_ioc, ca62_reg, ca64_reg);
ca713_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca713_ioc, ca71_reg, ca73_reg);
ca724_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca724_ioc, ca72_reg, ca74_reg);
ca813_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca813_ioc, ca81_reg, ca83_reg);
ca824_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca824_ioc, ca82_reg, ca84_reg);
ca913_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca913_ioc, ca91_reg, ca93_reg);
ca924_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca924_ioc, ca92_reg, ca94_reg);
ca1013_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca1013_ioc, ca101_reg, ca103_reg);
ca1024_in_reg: input_calos
	GENERIC MAP	(16)
	PORT MAP	(clk_inp,
		ca1024_ioc, ca102_reg, ca104_reg);
-- muons
mu1_in_reg: input_muons
	GENERIC MAP	(26)
	PORT MAP	(clk_inp,
		mu1_ioc, mu1_reg, mu2_reg);
mu3_in_reg: input_muons
	GENERIC MAP	(26)
	PORT MAP	(clk_inp,
		mu3_ioc, mu3_reg, mu4_reg);		

-- ************* CONDITION LOGIC *******************************
-- see condition-instantiations at end of file

-- ************* ALGORITHM LOGIC *******************************
-- logic of algo_and_or depends on definitions in def.xml, generated by gts !!!

call_algo_and_or: algo_and_or
	GENERIC	MAP(
			nr_ieg_4, nr_ieg_2_s, nr_ieg_2_wsc, nr_ieg_1_s, 
			nr_eg_4, nr_eg_2_s, nr_eg_2_wsc, nr_eg_1_s, 
			nr_jet_4, nr_jet_2_s, nr_jet_2_wsc, nr_jet_1_s, 
			nr_tau_4, nr_tau_2_s, nr_tau_2_wsc, nr_tau_1_s, 
			nr_fwdjet_4, nr_fwdjet_2_s, nr_fwdjet_2_wsc, nr_fwdjet_1_s,
			nr_muon_4, nr_muon_2_s, nr_muon_2_wsc, nr_muon_1_s, nr_muon_3,
			nr_jet_cnts_0_cond, nr_jet_cnts_1_cond, nr_jet_cnts_2_cond,
			nr_jet_cnts_3_cond, nr_jet_cnts_4_cond, nr_jet_cnts_5_cond,
			nr_jet_cnts_6_cond, nr_jet_cnts_7_cond, nr_jet_cnts_8_cond,
			nr_jet_cnts_9_cond, nr_jet_cnts_10_cond, nr_jet_cnts_11_cond,
			nr_ett_cond, nr_etm_cond, nr_htt_cond)
	PORT MAP(
		ieg_4, ieg_2_s, ieg_2_wsc, ieg_1_s, 
		eg_4, eg_2_s, eg_2_wsc, eg_1_s, 
		jet_4, jet_2_s, jet_2_wsc, jet_1_s, 
		tau_4, tau_2_s, tau_2_wsc, tau_1_s, 
		fwdjet_4, fwdjet_2_s, fwdjet_2_wsc, fwdjet_1_s, 
		muon_4, muon_2_s, muon_2_wsc, muon_1_s, muon_3, 
		jet_cnts_0_cond, jet_cnts_1_cond, jet_cnts_2_cond,
		jet_cnts_3_cond, jet_cnts_4_cond, jet_cnts_5_cond,
		jet_cnts_6_cond, jet_cnts_7_cond, jet_cnts_8_cond,
		jet_cnts_9_cond, jet_cnts_10_cond, jet_cnts_11_cond,
		ett_cond, etm_cond, htt_cond,
		ALGOSTROB_reg, algo_mux_in);

-- ************* INPUT DATA MUX *******************************
-- input data multiplexer part for tests

call_input_mux: in_mux_4_test
	PORT MAP(
		CLK40,
		ca11_reg, ca12_reg, ca13_reg, ca14_reg,
		ca21_reg, ca22_reg, ca23_reg, ca24_reg,
		ca31_reg, ca32_reg, ca33_reg, ca34_reg,
		ca41_reg, ca42_reg, ca43_reg, ca44_reg,
		ca51_reg, ca52_reg, ca53_reg, ca54_reg,
		ca61_reg, ca62_reg, ca63_reg, ca64_reg,
		ca71_reg, ca72_reg, ca73_reg, ca74_reg,
		ca81_reg, ca82_reg, ca83_reg, ca84_reg,
		ca91_reg, ca92_reg, ca93_reg, ca94_reg,
		ca101_reg, ca102_reg, ca103_reg, ca104_reg,
		mu1_reg, mu2_reg, mu3_reg, mu4_reg, 
		algo_mux_in,
		ENCOND, WRCOND,
		ADDR, VDATA(15 DOWNTO 0),
		DTACK_MUX,
		algo_cond_mem);

-- ************* CONDITION (ALGO) MEMORY *******************************
-- ALGO MEMORY AND TTC-SIGNALS INSTANTIATION
-- HB010206: algo-memory 6 x 1024 x 16 bit !!!!

-- mif_file : STRING := "algo_1024_16.mif"
-- mem_addr_cnt_width : integer := 10
-- mem_data_width : integer := 16
-- mem_width_base : integer := 3
-- mem_width : integer := 6
-- siehe cond_chip_pkg.vhd

call_cond_mem: cond_mem
	GENERIC MAP (algo_chip_name, mif_file, mem_addr_cnt_width,
				mem_data_width, mem_width_base, mem_width)
	PORT MAP(
		CLK_ALGO,
		algo_cond_mem,
		ENALGO, WRCOND,
		BCRES, L1A, L1RESET,
		ADDR, VDATA(15 DOWNTO 0),
		DTACK_COND_MEM,
		ALGO_OUT_REG_IN);

-- ************* ALGO OUT REGISTER *******************************
-- output-register
call_algo_out_reg: algo_out_reg
	GENERIC MAP(96, 3)
	PORT MAP(
		CLK_ALGO,
		ALGO_OUT_REG_IN, ALGOSTROB_reg,
		ALGO, ALGOSTROB);

-- ********************************************************************************************************
-- condition-instantiations parts:
-- RENAMING CALOs/MUONs
-- MUON CHARGE INSTANTIATIONS
-- CONDITION INSTANTIATIONS
-- ********************************************************************************************************

-- ************* RENAMING CALOs/MUONs *******************************
-- selecting and renaming of calorimeter- and muon-channels
-- fix structure see CMS-IN_02_069.PDF (HB, 080104) !!!
-- IEG inputs (channel 1)
reg_ieg_et_1 <= ca11_reg(5 DOWNTO 0);
reg_ieg_et_2 <= ca12_reg(5 DOWNTO 0);
reg_ieg_et_3 <= ca13_reg(5 DOWNTO 0);
reg_ieg_et_4 <= ca14_reg(5 DOWNTO 0);
reg_ieg_eta_1 <= ca11_reg(9 DOWNTO 6);
reg_ieg_eta_2 <= ca12_reg(9 DOWNTO 6);
reg_ieg_eta_3 <= ca13_reg(9 DOWNTO 6);
reg_ieg_eta_4 <= ca14_reg(9 DOWNTO 6);
reg_ieg_phi_1 <= ca11_reg(14 DOWNTO 10);
reg_ieg_phi_2 <= ca12_reg(14 DOWNTO 10);
reg_ieg_phi_3 <= ca13_reg(14 DOWNTO 10);
reg_ieg_phi_4 <= ca14_reg(14 DOWNTO 10);
reg_ieg_sync_1 <= ca11_reg(15);
reg_ieg_sync_2 <= ca12_reg(15);
reg_ieg_sync_3 <= ca13_reg(15);
reg_ieg_sync_4 <= ca14_reg(15);
-- EG inputs (channel 2)
reg_eg_et_1 <= ca21_reg(5 DOWNTO 0);
reg_eg_et_2 <= ca22_reg(5 DOWNTO 0);
reg_eg_et_3 <= ca23_reg(5 DOWNTO 0);
reg_eg_et_4 <= ca24_reg(5 DOWNTO 0);
reg_eg_eta_1 <= ca21_reg(9 DOWNTO 6);
reg_eg_eta_2 <= ca22_reg(9 DOWNTO 6);
reg_eg_eta_3 <= ca23_reg(9 DOWNTO 6);
reg_eg_eta_4 <= ca24_reg(9 DOWNTO 6);
reg_eg_phi_1 <= ca21_reg(14 DOWNTO 10);
reg_eg_phi_2 <= ca22_reg(14 DOWNTO 10);
reg_eg_phi_3 <= ca23_reg(14 DOWNTO 10);
reg_eg_phi_4 <= ca24_reg(14 DOWNTO 10);
reg_eg_sync_1 <= ca21_reg(15);
reg_eg_sync_2 <= ca22_reg(15);
reg_eg_sync_3 <= ca23_reg(15);
reg_eg_sync_4 <= ca24_reg(15);
--	cJET inputs (channel 3)
reg_jet_et_1 <= ca31_reg(5 DOWNTO 0);
reg_jet_et_2 <= ca32_reg(5 DOWNTO 0);
reg_jet_et_3 <= ca33_reg(5 DOWNTO 0);
reg_jet_et_4 <= ca34_reg(5 DOWNTO 0);
reg_jet_eta_1 <= ca31_reg(9 DOWNTO 6);
reg_jet_eta_2 <= ca32_reg(9 DOWNTO 6);
reg_jet_eta_3 <= ca33_reg(9 DOWNTO 6);
reg_jet_eta_4 <= ca34_reg(9 DOWNTO 6);
reg_jet_phi_1 <= ca31_reg(14 DOWNTO 10);
reg_jet_phi_2 <= ca32_reg(14 DOWNTO 10);
reg_jet_phi_3 <= ca33_reg(14 DOWNTO 10);
reg_jet_phi_4 <= ca34_reg(14 DOWNTO 10);
reg_jet_sync_1 <= ca31_reg(15);
reg_jet_sync_2 <= ca32_reg(15);
reg_jet_sync_3 <= ca33_reg(15);
reg_jet_sync_4 <= ca34_reg(15);
--	fwdJET inputs (channel 4)
reg_fwdjet_et_1 <= ca41_reg(5 DOWNTO 0);
reg_fwdjet_et_2 <= ca42_reg(5 DOWNTO 0);
reg_fwdjet_et_3 <= ca43_reg(5 DOWNTO 0);
reg_fwdjet_et_4 <= ca44_reg(5 DOWNTO 0);
reg_fwdjet_eta_1 <= ca41_reg(9 DOWNTO 6);
reg_fwdjet_eta_2 <= ca42_reg(9 DOWNTO 6);
reg_fwdjet_eta_3 <= ca43_reg(9 DOWNTO 6);
reg_fwdjet_eta_4 <= ca44_reg(9 DOWNTO 6);
reg_fwdjet_phi_1 <= ca41_reg(14 DOWNTO 10);
reg_fwdjet_phi_2 <= ca42_reg(14 DOWNTO 10);
reg_fwdjet_phi_3 <= ca43_reg(14 DOWNTO 10);
reg_fwdjet_phi_4 <= ca44_reg(14 DOWNTO 10);
reg_fwdjet_sync_1 <= ca41_reg(15);
reg_fwdjet_sync_2 <= ca42_reg(15);
reg_fwdjet_sync_3 <= ca43_reg(15);
reg_fwdjet_sync_4 <= ca44_reg(15);
--	TAU inputs (channel 5)
reg_tau_et_1 <= ca51_reg(5 DOWNTO 0);
reg_tau_et_2 <= ca52_reg(5 DOWNTO 0);
reg_tau_et_3 <= ca53_reg(5 DOWNTO 0);
reg_tau_et_4 <= ca54_reg(5 DOWNTO 0);
reg_tau_eta_1 <= ca51_reg(9 DOWNTO 6);
reg_tau_eta_2 <= ca52_reg(9 DOWNTO 6);
reg_tau_eta_3 <= ca53_reg(9 DOWNTO 6);
reg_tau_eta_4 <= ca54_reg(9 DOWNTO 6);
reg_tau_phi_1 <= ca51_reg(14 DOWNTO 10);
reg_tau_phi_2 <= ca52_reg(14 DOWNTO 10);
reg_tau_phi_3 <= ca53_reg(14 DOWNTO 10);
reg_tau_phi_4 <= ca54_reg(14 DOWNTO 10);
reg_tau_sync_1 <= ca51_reg(15);
reg_tau_sync_2 <= ca52_reg(15);
reg_tau_sync_3 <= ca53_reg(15);
reg_tau_sync_4 <= ca54_reg(15);
--	energy summary information ("single objects") inputs (channel 6)
reg_ett <= ca61_reg(12 DOWNTO 0);
reg_esums_sync_1 <= ca61_reg(15);
reg_etm <= ca62_reg(12 DOWNTO 0);
reg_esums_sync_2 <= ca62_reg(15);
reg_htt <= ca63_reg(12 DOWNTO 0);
reg_esums_sync_3 <= ca63_reg(15);
reg_etm_phi <= ca64_reg(6 DOWNTO 0);
reg_esums_sync_4 <= ca64_reg(15);
--	jet counts inputs (channel 7)
reg_jet_cnts_0 <= ca71_reg(4 DOWNTO 0);
reg_jet_cnts_1 <= ca71_reg(9 DOWNTO 5);
reg_jet_cnts_2 <= ca71_reg(14 DOWNTO 10);
reg_jet_cnts_sync_1 <= ca71_reg(15);
reg_jet_cnts_3 <= ca72_reg(4 DOWNTO 0);
reg_jet_cnts_4 <= ca72_reg(9 DOWNTO 5);
reg_jet_cnts_5 <= ca72_reg(14 DOWNTO 10);
reg_jet_cnts_sync_2 <= ca72_reg(15);
reg_jet_cnts_6 <= ca73_reg(4 DOWNTO 0);
reg_jet_cnts_7 <= ca73_reg(9 DOWNTO 5);
reg_jet_cnts_8 <= ca73_reg(14 DOWNTO 10);
reg_jet_cnts_sync_3 <= ca73_reg(15);
reg_jet_cnts_9 <= ca74_reg(4 DOWNTO 0);
reg_jet_cnts_10 <= ca74_reg(9 DOWNTO 5);
reg_jet_cnts_11 <= ca74_reg(14 DOWNTO 10);
reg_jet_cnts_sync_4 <= ca74_reg(15);
--	MUON inputs
reg_muon_pt_1 <= mu1_reg(12 DOWNTO 8);
reg_muon_pt_2 <= mu2_reg(12 DOWNTO 8);
reg_muon_pt_3 <= mu3_reg(12 DOWNTO 8);
reg_muon_pt_4 <= mu4_reg(12 DOWNTO 8);
reg_muon_eta_1 <= mu1_reg(21 DOWNTO 16);
reg_muon_eta_2 <= mu2_reg(21 DOWNTO 16);
reg_muon_eta_3 <= mu3_reg(21 DOWNTO 16);
reg_muon_eta_4 <= mu4_reg(21 DOWNTO 16);
reg_muon_phi_1 <= mu1_reg(7 DOWNTO 0);
reg_muon_phi_2 <= mu2_reg(7 DOWNTO 0);
reg_muon_phi_3 <= mu3_reg(7 DOWNTO 0);
reg_muon_phi_4 <= mu4_reg(7 DOWNTO 0);
reg_muon_qual_1 <= mu1_reg(15 DOWNTO 13);
reg_muon_qual_2 <= mu2_reg(15 DOWNTO 13);
reg_muon_qual_3 <= mu3_reg(15 DOWNTO 13);
reg_muon_qual_4 <= mu4_reg(15 DOWNTO 13);
reg_muon_iso_1 <= mu1_reg(22);
reg_muon_iso_2 <= mu2_reg(22);
reg_muon_iso_3 <= mu3_reg(22);
reg_muon_iso_4 <= mu4_reg(22);
reg_muon_mip_1 <= mu1_reg(23);
reg_muon_mip_2 <= mu2_reg(23);
reg_muon_mip_3 <= mu3_reg(23);
reg_muon_mip_4 <= mu4_reg(23);
reg_muon_sy_0_1 <= mu1_reg(24);
reg_muon_sy_0_2 <= mu2_reg(24);
reg_muon_sy_0_3 <= mu3_reg(24);
reg_muon_sy_0_4 <= mu4_reg(24);
reg_muon_sy_1_1 <= mu1_reg(25);
reg_muon_sy_1_2 <= mu2_reg(25);
reg_muon_sy_1_3 <= mu3_reg(25);
reg_muon_sy_1_4 <= mu4_reg(25);

-- ************* MUON CHARGE INSTANTIATIONS *******************************
-- muon charge instantiations, always instantiated
-- compiler reduces unused logic

-- muon sync_word NOT used in VHDL Version 6.x

mu_ch_1: m_charge_cond_1
PORT MAP (
reg_muon_sy_0_1, reg_muon_sy_1_1,
reg_muon_sy_0_2, reg_muon_sy_1_2,
reg_muon_sy_0_3, reg_muon_sy_1_3,
reg_muon_sy_0_4, reg_muon_sy_1_4,
pos_lut_1, pos_lut_2, pos_lut_3, pos_lut_4,
neg_lut_1, neg_lut_2, neg_lut_3, neg_lut_4);

mu_ch_2: m_charge_cond_2
PORT MAP (
reg_muon_sy_0_1, reg_muon_sy_1_1,
reg_muon_sy_0_2, reg_muon_sy_1_2,
reg_muon_sy_0_3, reg_muon_sy_1_3,
reg_muon_sy_0_4, reg_muon_sy_1_4,
eq_lut_12, eq_lut_13, eq_lut_14,
eq_lut_23, eq_lut_24, eq_lut_34,
neq_lut_12, neq_lut_13, neq_lut_14,
neq_lut_23, neq_lut_24, neq_lut_34);

mu_ch_3: m_charge_cond_3
PORT MAP (
reg_muon_sy_0_1, reg_muon_sy_1_1,
reg_muon_sy_0_2, reg_muon_sy_1_2,
reg_muon_sy_0_3, reg_muon_sy_1_3,
reg_muon_sy_0_4, reg_muon_sy_1_4,
eq_lut_123, eq_lut_124, eq_lut_134, eq_lut_234,
neq_lut_123, neq_lut_124, neq_lut_134, neq_lut_234);

mu_ch_4: m_charge_cond_4
PORT MAP (
reg_muon_sy_0_1, reg_muon_sy_1_1,
reg_muon_sy_0_2, reg_muon_sy_1_2,
reg_muon_sy_0_3, reg_muon_sy_1_3,
reg_muon_sy_0_4, reg_muon_sy_1_4,
eq_lut_1234, pair_lut);

-- ************* CONDITION INSTANTIATIONS *******************************
-- variable part, depends on definitions in def.xml, generated by gts !!!

$(calo_common)

$(esums_common)

$(jet_cnts_common)

$(muon_common)

$(calo)

$(esums)

$(jet_cnts)

$(muon)

END ARCHITECTURE rtl;
