#!/bin/sh

guidir=/data/ecalod-disk01/dqm-gui
srcdir=/nfshome0/ecalpro/DQM/gui

manage="$guidir/current/config/dqmgui/manage-ecal -f ecal"

dqmenv(){
    if [ ${HOSTNAME} != "srv-S2F19-29" ]; then
	echo "This is not ecalod-disk01 !!!"
	exit 1
    fi
    cd $guidir
    . $PWD/current/apps/dqmgui/etc/profile.d/env.sh
}

dqmdeploy(){
    export http_proxy=http://cmsproxy.cms:3128/

    cd $guidir
    $PWD/cfg/Deploy -R cmsweb@1203h -t MYDEV -s "prep sw post" $PWD dqmgui/bare
    cp $srcdir/scripts/manage-ecal $PWD/current/config/dqmgui/
}

ulimit -c 0

case "$1" in
    start)
	dqmdeploy
	dqmenv
	$manage start "I did read documentation"
	;;
    stop)
        dqmenv
	$manage stop "I did read documentation"
	;;
    update)
	cd $srcdir/cfg
	svn up
	;;
    *)
	echo "Usage: guiControl.sh (start|stop|update)"
	exit 1
	;;
esac

exit 0
