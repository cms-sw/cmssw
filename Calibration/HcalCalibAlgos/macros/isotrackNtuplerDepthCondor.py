#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uproot


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit HTCondor jobs for isotrack ntuples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core required paths
    parser.add_argument("--filePU", dest="PU_file", type=str, required=True, help="Input PU ROOT file")
    parser.add_argument("--fileNPU", dest="NoPU_file", type=str, required=True, help="Input NoPU ROOT file")
    parser.add_argument("--isotrackNtupler", dest="IsotrackNtupler", type=str, required=True, help="Path to isotrackNtupler script isotrackNtupler_depth_vF.py")
    parser.add_argument("-o", "--output", dest="output_dir", type=str, required=True, help="Output directory for parquets")

    # Condor & execution controls
    parser.add_argument("-q", "--flavour", dest="jobFlavour", type=str, default="workday", help="HTCondor job flavour")
    parser.add_argument("--job-dir", dest="job_dir", type=str, default="condor_jobs", help="Directory to store .sub/.out/.err files")
    parser.add_argument("--python", dest="python_bin", type=str,
                        default="/eos/user/d/dasgupsu/anaconda3/envs/dna/bin/python3.8",
                        help="Path to Python executable in Condor")
    parser.add_argument("--dryRun", action="store_true", help="Generate submit scripts without calling condor_submit")

    # Ntupler parameters to forward
    parser.add_argument("-t", "--tree", default="hcalIsoTrkAnalyzer/CalibTree", help="TTree name inside ROOT files")
    parser.add_argument("--events-per-job", type=int, default=50000, help="Number of PU events to process per job")
    parser.add_argument("--max-depth", type=int, default=7, help="Max HCAL depth to pass to ntupler")
    parser.add_argument("--scale", type=int, default=500000, help="No-PU chunk size per iteration inside ntupler")

    args = parser.parse_args()

    # Ensure output and job directories exist
    job_dir_path = Path(args.job_dir).resolve()
    output_dir_path = Path(args.output_dir).resolve()
    job_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading PU tree '{args.tree}' from: {args.PU_file}")
    with uproot.open(args.PU_file) as pu_root:
        pu_tree = pu_root[args.tree]
        pu_entries = getattr(pu_tree, "num_entries", getattr(pu_tree, "numentries", None))

    job_files = []
    job_idx = 0

    for index in range(0, pu_entries, args.events_per_job):
        pu_start = index
        pu_stop = min(index + args.events_per_job, pu_entries)

        output_file_stem = f"{output_dir_path}/IsotrackNtupler"
        cmd_args = (
            f"{args.IsotrackNtupler} "
            f"-PU {args.PU_file} "
            f"-NPU {args.NoPU_file} "
            f"-s {pu_start} "
            f"-e {pu_stop} "
            f"-O {output_file_stem} "
            f"-t {args.tree} "
            f"--max-depth {args.max_depth} "
            f"--scale {args.scale}"
        )

        job_prefix = f"RootTreeMaker_{job_idx}_{pu_start}_{pu_stop}"
        job_file_sub = job_dir_path / f"{job_prefix}.sub"
        job_file_out = job_dir_path / f"{job_prefix}.out"
        job_file_err = job_dir_path / f"{job_prefix}.err"
        job_file_log = job_dir_path / f"{job_prefix}.log"

        submit_content = (
            f"executable = {args.python_bin}\n"
            f"arguments = {cmd_args}\n"
            f"output = {job_file_out}\n"
            f"error = {job_file_err}\n"
            f"log = {job_file_log}\n"
            f'transfer_output_files = ""\n'
            f"getenv = True\n"
            f'+JobFlavour = "{args.jobFlavour}"\n'
            f"queue 1\n"
        )

        with open(job_file_sub, "w") as sf:
            sf.write(submit_content)

        job_files.append(str(job_file_sub))
        job_idx += 1

    print(f"Generated {len(job_files)} submit descriptions in: {job_dir_path}")

    if args.dryRun:
        print("Dry run requested; skipping submission.")
        return

    print("Submitting jobs to HTCondor...")
    for jf in job_files:
        cmd = ["condor_submit", "-spool", jf] if str(job_dir_path).startswith("/eos") else ["condor_submit", jf]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error submitting {jf}:\n{res.stderr}")
        else:
            print(res.stdout.strip())


if __name__ == "__main__":
    main()
