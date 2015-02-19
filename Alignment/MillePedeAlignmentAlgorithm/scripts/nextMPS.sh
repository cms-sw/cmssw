#!/bin/sh

#    Script will be able to do next things: 
# -- run MILLE, PEDE or MILLEPEDE jobs 
# -- check out for FAILs and resubmit it
# -- send an email when MILLE, PEDE or MILLEPEDE jobs are ready
# you can setup time delay in FAILs checking (-t)
# plese setup mail adress to get INFO about jobs status

help(){
echo "*************************************************************************************"
                        echo " "
                        echo " Options:"
                        echo "-h, --help                show brief help"
			echo "-n, 			specify N mille jobs, default is Njobs = 429"
			echo "-m,			run MILLE jobs only"
			echo "-p,			run PEDE  jobs only"
			echo "-mp, 			run MILLEPEDE jobs"
			echo "-email, 			specify you e-mail adress, or manually change it in the script"
			echo "-t,			specify time of script sleeping, Example : -t 10m, -t 20s"
			echo ""
			echo "  Run example: sh nextMPS.sh -n 429 -m -email uname@gmail.com -t 5m"
			echo " "
echo "*************************************************************************************"
                        exit 0
}

if [ "$#" == "0" ]; then
        echo "This script needs arguments. Please use in this way:"
	help
        exit 1
fi

Njobs=429
echo "*************************************************************************************"

while (("$#")); do
        case "$1" in
                -h|--help)
                        help
                        exit 0
                        ;;
                -n)
                        shift
                        if (("$#")); then
				Njobs=$1
				echo "You specified Njobs = $Njobs"
                        else
                                echo "Number of jobs is not specified. Please use -h, --help"
                                exit 1
                        fi
                        shift
                        ;;
                -m)
                        if (("$#")); then
        			echo "You will run MILLE jobs only"
				runstatus="m"
				#echo "Runstatus = $runstatus"
                        fi
                        shift
                        ;;
                -p)
                        if (("$#")); then
                                echo "You will run PEDE jobs only"
                                runstatus="p"
                                #echo "Runstatus = $runstatus"
                        fi
                        shift
                        ;;
		-mp)
                        if (("$#")); then
                                echo "You will run MILLEPEDE jobs"
                                runstatus="mp"
                                #echo "Runstatus = $runstatus"
                        fi
                        shift
                        ;;
                -email)
			shift
                        if (("$#")); then
                                echo "Job status will be send to the email adress: $1"
                                email="$1"
                               # echo "Email = $email"
				else 
				email="0"
				echo "You did not specified e-mail adress"
                        fi
                        shift
                        ;;

	        -t)
                        shift
                        if (("$#")); then
                                TimeSleep=$1
				time_flag="spec"
                                echo "You specified time script sleeping = $1"
                        else
                                echo "TimeSleep is not specified. Please use -h, --help"
                                exit 1
                        fi
                        shift
                        ;;
					
                *)
                        break
                        ;;
        esac
done

echo "*************************************************************************************"
if [ "$runstatus" = "m" ] || [ "$runstatus" = "mp" ]; then
	if [ ! -f nextMPS.log  ]; then
    	echo "nextMPS.log will be created"
	else
    	rm nextMPS.log	
    	echo "Old nextMPS.log file removed"
	fi
fi

TESTJOBS(){
# testing your Jobs for FAIL status and resubmitting

while [ "$(mps_stat.pl | grep OK | wc -l)" != "$Njobs" ]
do	
	if [ "$time_flag" = "spec" ]; then
        	sleep $TimeSleep
		else 
		sleep 10m
	fi

        echo "Fetch DONE jobs"
        mps_fetch.pl

        if [ $(mps_stat.pl | grep FAIL | wc -l) -ne 0 ];
        then
                Njfails=$(echo "$(mps_stat.pl | grep FAIL | wc -l)")
                echo "Njob fails = $Njfails"
                echo "Retry failed jobs"
                mps_retry.pl FAIL
                mps_fire.pl $Njfails
        else
                NjOK=$(mps_stat.pl | grep OK | wc -l)
                echo " Checking jobs . Njobs OK = $NjOK"

        fi
done
}


MILLERUN(){
# Setup your folder for MILLEPEDE run. Job status after: PEND
	./setup_align.pl 

# Send your jobs for running. Job status after: RUN
	mps_fire.pl $Njobs

# testing your MILLE jobs for FAIL status and resubmitting
	TESTJOBS

if [ "$email" != "0" ]; then
(echo "Subject: MILLE jobs ready"; echo "Your MILLE jobs are ready. Njobs mille OK = $NjOK";) | sendmail $email
else echo "You did not setup e-mail"
fi

	}

PEDERUN(){
	mps_fire.pl -m
	Njobs=$[Njobs+1]
	TESTJOBS

if [ "$email" != "0" ]; then
(echo "Subject: PEDE jobs ready"; echo "Your PEDE jobs are ready. Njobs pede OK = 1";) | sendmail $email
else echo "You did not setup e-mail"
fi

	}

MILLEPEDERUN()	{
	MILLERUN
	PEDERUN
		}

# Run part:
            	if [ "$runstatus" = "m" ]; then
      		echo "MILLE jobs will be run"
		MILLERUN >> nextMPS.log          
            	else
			if [ "$runstatus" = "p" ]; then
                	echo "PEDE jobs will be run"
			PEDERUN >> nextMPS.log
			else 
				if [ "$runstatus" = "mp" ]; then
				echo "MILLEPEDE jobs will be run"
				MILLEPEDERUN >> nextMPS.log
				fi
			fi

            	fi	
# END SCRIPT
echo "THE END"

