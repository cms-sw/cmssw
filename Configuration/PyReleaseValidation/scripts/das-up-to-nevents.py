#!/usr/bin/env python3
import subprocess
import sys
import argparse


def das_file_nevents(X):
    cmd = "dasgoclient --query 'file=%s | grep file.nevents'"%(X.replace("\n",""))
    return int(subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode('utf8'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--list','-l', default=None, help="List of files",nargs="+",type=str)
    parser.add_argument('--input','-i', default=None, help="File with e list of files",type=str)
    parser.add_argument('--threshold','-t', help ="Event threshold per file",type=int,default=-1)
    parser.add_argument('--events','-e', help ="Tot number of events targeted",type=int,default=-1)
    parser.add_argument('--outfile','-o', help='Dump results to file', type=str, default=None)
    args = parser.parse_args()

    if args.threshold < 0 and args.events <0:
        print("Please set at least one of --events or --threshold > 0")
        sys.exit(1)

    N = 0
    files = []

    if args.list is not None and args.input is not None:
        print("Choose one. Either:")
        print("- a list of files with --list")
        print("- an input file with a list of samples with --input")
        sys.exit(1)

    if args.input is not None:
        with open(args.input,"r") as f:
            for line in f:
                F = line.replace("\n","")
                n = das_file_nevents(F)
                if n < args.threshold: # skip small files
                    continue
                files.append(F)
                N = N + n
                if N >= args.events: # in excess
                    break
    elif args.list is not None:
        for F in args.list:
            n = das_file_nevents(F)
            if n < args.threshold: # skip small files
                continue
            files.append(F)
            N = N + n
            if N >= args.events: # in excess
                break
    
    if args.outfile is not None and N>0:
        with open(args.outfile, 'w') as f:
            for line in files:
                f.write(f"{line}\n") 
    else:
        print("\n".join(files))

    sys.exit(0)
    
