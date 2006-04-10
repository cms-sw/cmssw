#!/bin/bash

sqlplus -S cond01/oracond01@ecalh4db < dump_monitoring.sql | less -S