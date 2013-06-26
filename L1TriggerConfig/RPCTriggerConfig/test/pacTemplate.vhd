XXV

library ieee;
use ieee.std_logic_1164.all;
use work.std_logic_1164_ktp.all;
use work.RPC_PAC_def.all;

package RPC_PAC_patt is
constant TT_MPAC_CFG_ID         :THV(3 downto 0) := "XXXX";
constant TT_MPAC_OR_LIMIT       :natural := 0; --| 0...3
XXN
constant TT_GBSORT_INPUTS       :natural := 12;

constant PACLogPlainsDecl       :TPACLogPlainsDecl := (
  --PAC_INDEX
  --|   PAC_MODEL
  --|   |      logplane 1 size .........logplane 6 size
XXP

constant LogPlainConn           :TLogPlainConn := (
  --PAC_INDEX   Logplane        LinkChannel     LinkLeftBit
  --| PAC_MODEL |       Link    |       LogPlaneLeftBit
  --|      |    |       |       |       |       |       LinkBitsCount
  --------------------------------------------------------------
XXC

constant PACCellQuality :TPACCellQuality := (
XXQ


constant PACPattTable :TPACPattTable := (
--PAC_INDEX
--| PAC_MODEL 
--| | Ref Group Index
--| | | Qualit Tab index
--| | | |  Plane1  Plane2  Plane3  Plane4  Plane5  Plane6  sign code  pat number
XXS



constant GBSortDecl		:TGBSortDecl := (  
 --PAC_INDEX
 --|   PAC_MODEL
 --|       |   GBSORT_INPUT_INDEX
XXG

end RPC_PAC_patt;
