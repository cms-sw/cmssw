#!/usr/bin/env python3
from typing import List, Tuple
from datetime import datetime
import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import os
import sys

process = cms.Process('compareLHCInfosToPPSDB')

options = VarParsing.VarParsing()
options.register( 'ppsFile'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "file with the pps data in the format: \
                (FILL_NUMBER, RUN_NUMBER, LUMI_SECTION, BETA_STAR_P5_X_M, XING_ANGLE_P5_X_URAD, DIP_UPDATE_TIME as 'DD/MM/YYYY HH24:MI:SS') "
                  )
options.register( 'LHCInfoFile'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "file with the LHCInfo data in the format: TODO "
                  )
options.register( 'PerLSFile'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "file with the LHCInfoPerLS data in the format: TODO "
                  )
options.register( 'separator'
                , ','
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "separator of the input CSV files"
                  )
options.register( 'debug'
                , False
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.bool
                , "debug mode, prints additional information"
                  )
options.parseArguments()


def to_unix_timestamp(date_string):
    date_format = "%d/%m/%Y %H:%M:%S"
    datetime_obj = datetime.strptime(date_string, date_format)
    unix_timestamp = int(time.mktime(datetime_obj.timetuple()))
    return unix_timestamp


def to_date_string(unix_timestamp):
    datetime_obj = datetime.fromtimestamp(unix_timestamp)
    date_string = datetime_obj.strftime("%d/%m/%Y %H:%M:%S")
    return date_string

def read_split(file) -> Tuple[str, List[str]]:
    line = file.readline()
    return line, line.split(options.separator)

def main():
    process_lhc = options.LHCInfoFile != ""
    process_perls = options.PerLSFile != ""

    #TODO check if at least one of the lhcinfo files were provided

    f_pps = open(options.ppsFile)
    f_lhc = open(options.LHCInfoFile) if process_lhc else None
    f_perls = open(options.PerLSFile) if process_perls else None
    
    print("fill", "pps run", "pps LS", "pps beta*", "pps xangle", "pps dip date", sep = ";", end="")
    if process_lhc:
        print("record", "LS LHCInfo", "beta* LHCInfo", "xangle LHCInfo", "delivLumi LHCInfo", "IOV LHCInfo", "IOV date LHCInfo",
            "beta* |pps-LHCInfo| error", "xangle |pps-LHCInfo| error", sep = ";", end="")
    if process_perls:
        print("record", "LS PerLS", "beta*X", "beta*Y", "xangleX", "xangleY", "IOV PerLS", "IOV date PerLS",
            "beta*X |pps-PerLS| error", "xangleX |pps-PerLS| error", sep = ";", end="")

    print()

    lhc_beta = None
    lhc_xangle = None
    perls_beta = None
    perls_xangle = None

    if process_lhc:
        line_lhc, split_lhc = read_split(f_lhc)
        #skip the headers
        if "IOV" in line_lhc or line_lhc == "\n":
            if options.debug:
                print("skipping the LHCInfo header")
            line_lhc, split_lhc = read_split(f_lhc)

    if process_perls:
        line_perls, split_perls = read_split(f_perls)
        #skip the headers
        if "IOV" in line_perls or line_perls == "\n":
            if options.debug:
                print("skipping the LHCInfoPerLS header")
            line_perls, split_perls = read_split(f_perls)


    for line_pps in f_pps:
        line_pps = line_pps[1:-2]
        split_pps = line_pps.split(', ')

        for i, val in enumerate(split_pps):
            print(val, end = ";" if i <= 5 or i == len(split_pps)-1 else ", ")
        

        print_lhc = False
        print_perls = False
        # print("  pps:", (split_pps[2]), "lhc:", (split_lhc[2]))
        if process_lhc:
            if len(split_pps) > 5 and len(split_lhc) > 5:
                # if len(split_pps[5][1:-1]) != len('17/04/2023 07:30:42'):
                #     print(split_pps[5][1:-1], len(split_pps[5][1:-1]), len('17/04/2023 07:30:42'))
                #     print(split_pps[5][1:-1])
                #     print('17/04/2023 07:30:42')
                #     exit(0)
                pps_timestamp = to_unix_timestamp(split_pps[5][1:-1]) + 2*60*60 # - coversion of time zones
                # print(pps_timestamp, "-", int(split_lhc[2]), "=", pps_timestamp -int(split_lhc[2]), end = " |; ")
                if(abs(pps_timestamp-int(split_lhc[2])) < 10*60 and split_pps[2] == split_lhc[3]):
                    # print(line_lhc, end="")
                    print(split_lhc[1], split_lhc[3], split_lhc[5], split_lhc[4], split_lhc[6],
                        split_lhc[0], to_date_string(int(split_lhc[2])), sep=";", end=";" + "" if process_perls else "\n")
                    while(len(split_lhc) > 5 and abs(pps_timestamp-int(split_lhc[2])) < 10*60 and split_pps[2] == split_lhc[3]):
                        lhc_beta = float(split_lhc[5])
                        lhc_xangle = float(split_lhc[4])
                        line_lhc, split_lhc = read_split(f_lhc)
                else:
                    # print(";;;;;;;", end = "")
                    print(";;;;;;;                                                            ", end = "" if process_perls else "\n")
                    print_lhc = True
                
            #                 lhc_beta = lhc_beta
            # lhc_xangle = lhc_xangle
                pps_beta = float(split_pps[3])
                pps_xangle = float(split_pps[4])
                # print(pps_beta, lhc_beta, pps_xangle, lhc_xangle, sep=" - ", end = " | ")
                print(abs(pps_beta-lhc_beta) if lhc_beta is not None else 0,
                    abs(pps_xangle-lhc_xangle) if lhc_xangle is not None else 0, sep = ";", end = ";" + "" if process_perls else "\n")
            else:
                print(";;;;;;;;;", end = "" if process_perls else "\n")

        if process_perls:
            time_ix_ls = 2
            ls_ix_ls = 4
            run_ix_ls = 3
            beta_ix_ls = 7
            xangle_ix_ls = 5
            if(len(split_pps) > 5 and len(split_perls) > max(time_ix_ls, ls_ix_ls, run_ix_ls, beta_ix_ls, xangle_ix_ls)):
                if(split_pps[1] == split_perls[run_ix_ls] and split_pps[2] == split_perls[ls_ix_ls]):
                    # print(line_perls, end="")
                    print(split_perls[1], split_perls[4], split_perls[7], split_perls[8], split_perls[5], split_perls[6],
                        split_perls[0], to_date_string(int(split_perls[time_ix_ls])), sep=";", end=";")
                    perls_beta = float(split_perls[beta_ix_ls])
                    perls_xangle = float(split_perls[xangle_ix_ls])

                    line_perls, split_perls = read_split(f_perls)
                    # print("next line: " , line_perls)
                else:
                    # print(";;;;;;;", end = "")
                    # print("run and ls mismatch: " , line_perls)
                    print(";;;;;;;;                                                            ", end = "")
                    print_perls = True
                
            #                 perls_beta = perls_beta
            # perls_xangle = perls_xangle
                pps_beta = float(split_pps[3])
                pps_xangle = float(split_pps[4])
                # print(pps_beta, perls_beta, pps_xangle, perls_xangle, sep=" - ", end = " | ")
                print(abs(pps_beta-perls_beta) if perls_beta is not None else 0,
                    abs(pps_xangle-perls_xangle) if perls_xangle is not None else 0, sep = ";", end = ";\n")
            else:
                print(";;;;;;;;;;")#, end = "")
                if options.debug:
                    print("not enough fields", line_perls)

    f_pps.close()
    if process_lhc:
        f_lhc.close() 
    if process_perls:
        f_perls.close() 
        
        

if __name__ == "__main__":
    main()
