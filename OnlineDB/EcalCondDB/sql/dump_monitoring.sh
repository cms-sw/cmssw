#!/bin/bash

sqlplus -S test04/oratest04@ecalh4db < dump_monitoring.sql | less -S