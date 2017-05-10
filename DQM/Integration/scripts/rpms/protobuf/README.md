# Protobuf RPM
I am documenting how to build a `RPM` for Google Protocol Buffer version 2.4.1 in EL6.
I could not have succeeded in doing that without the [Fedora RPM guide](http://fedoraproject.org/wiki/How_to_create_an_RPM_package),
as well as the references therein.
Please read it: it is useful!

## Getting the sources
The starting point is the Fedora `RPM` for version 2.4.1, documented [here](https://admin.fedoraproject.org/pkgdb/acls/name/protobuf).
You can see all the development branches in the git repository for the package:
just click on package source on the top or go directly [here](http://pkgs.fedoraproject.org/cgit/protobuf.git/).

In the git log GUI, find the last commit for version 2.4.1 (it is the tip of `f17` and `f18` branches),
and download the snapshot (use `-N` with `wget`, `-R` with `curl` in order to preserve timestamping of files):

    $ wget -N http://pkgs.fedoraproject.org/cgit/protobuf.git/snapshot/protobuf-b2c5b375e8c0c198e10bbdfeeda9afa69363c0dc.tar.gz

Untar it:

    $ tar -xvzf protobuf-b2c5b375e8c0c198e10bbdfeeda9afa69363c0dc.tar.gz

The following files in there:

    ftdetect-proto.vim
    protobuf-2.3.0-fedora-gtest.patch
    protobuf-2.4.1-java-fixes.patch
    protobuf-init.el

are in this repo, in the `SOURCES` subdirectory.

If you want to build the `RPM` by yourself, create the `RPM` building directory structure

    $ rpmdev-setuptree

and copy them (`cp -p` preserves timestamps) into `~/rpmbuild/SOURCES`.

Finally, download the protocol buffer archive into the `~/rpmbuild/SOURCES` folder:

    $ wget -N http://protobuf.googlecode.com/files/protobuf-2.4.1.tar.bz2

## Getting the spec file
The inspiration for building the spec file comes from the one packed into
[this source RPM](http://yum.aclub.net/pub/linux/centos/6/umask/SRPMS/protobuf-2.4.1-1.el6.src.rpm).
The spec file is in the `SPECS` subdirectory of this repo.
Give a look at it, and check the differences w.r.t. the original Fedora file:
you can see that I have put a condition for building the emacs and emacs-el packages:
EL5 and EL6 cannot build them, so they must be skipped.

There are some other conditions specific to EL5, due to the fact that some of the RPM macros are not supported in such arch.
Even if the package builds, I cannot guarantee (at the moment) it is fully functional: you are welcome to help!
If you want to build the `RPM` by yourself, copy the spec file into `~/rpmbuild/SPECS`.

## Building the RPM
Note: Use a local dummy user for building RPMs! See the [Fedora RPM guide](http://fedoraproject.org/wiki/How_to_create_an_RPM_package) for more information.

Go into the `~rpmbuild/SPECS` directory, and validate the spec file:

    $ rpmlint protobuf.spec

You should not get any errors (if so, I made something very wrong), but you should get a warning for the URL of the source:

    protobuf.spec: W: invalid-url Source0: http://protobuf.googlecode.com/files/protobuf-2.4.1.tar.bz2 HTTP Error 404: Not Found

It is a false positive (see the [bug report](https://bugzilla.redhat.com/show_bug.cgi?id=767739)).

Now, you can build the `RPM`:

    $ rpmbuild -ba -vv protobuf.spec

If the buildas succeeds, you can find the `RPM` in `~/rpmbuild/RPMS`, and the source `RPM` in `~/rpmbuild/SRPMS`.

### Special steps for EL5
Unfortunately, the distribution macro is not supported under RHEL5 (see [here](http://stackoverflow.com/questions/5135502/rpmbuild-dist-not-defined-on-centos-5-5)). In order to fix that, you should install, as root, the package `buildsys-macros`:

    # yum install buildsys-macros

You will also need to install python-setuptools:

    # yum install python-setuptools

Unfortunately (I have not investigated further, so you are welcome to help), the RPM building fails during the install step because of rpaths (see the [Fedora packaging guidelines](https://fedoraproject.org/wiki/Packaging:Guidelines) for an explanation, and just google `rpath` if you want to learn about dynamic linking in Linux) with the following message:

    + /usr/lib/rpm/check-rpaths /usr/lib/rpm/check-buildroot
    *******************************************************************************
    *
    * WARNING: 'check-rpaths' detected a broken RPATH and will cause 'rpmbuild'
    *          to fail. To ignore these errors, you can set the '$QA_RPATHS'
    *          environment variable which is a bitmask allowing the values
    *          below. The current value of QA_RPATHS is 0x0000.
    *
    *    0x0001 ... standard RPATHs (e.g. /usr/lib); such RPATHs are a minor
    *               issue but are introducing redundant searchpaths without
    *               providing a benefit. They can also cause errors in multilib
    *               environments.
    *    0x0002 ... invalid RPATHs; these are RPATHs which are neither absolute
    *               nor relative filenames and can therefore be a SECURITY risk
    *    0x0004 ... insecure RPATHs; these are relative RPATHs which are a
    *               SECURITY risk
    *    0x0008 ... the special '$ORIGIN' RPATHs are appearing after other
    *               RPATHs; this is just a minor issue but usually unwanted
    *    0x0010 ... the RPATH is empty; there is no reason for such RPATHs
    *               and they cause unneeded work while loading libraries
    *    0x0020 ... an RPATH references '..' of an absolute path; this will break
    *               the functionality when the path before '..' is a symlink
    *          
    *
    * Examples:
    * - to ignore standard and empty RPATHs, execute 'rpmbuild' like
    *   $ QA_RPATHS=$[ 0x0001|0x0010 ] rpmbuild my-package.src.rpm
    * - to check existing files, set $RPM_BUILD_ROOT and execute check-rpaths like
    *   $ RPM_BUILD_ROOT=<top-dir> /usr/lib/rpm/check-rpaths
    *  
    *******************************************************************************
    ERROR   0001: file '/usr/lib64/libprotoc.so.7.0.0' contains a standard rpath '/usr/lib64' in [/usr/lib64]
    ERROR   0001: file '/usr/bin/protoc' contains a standard rpath '/usr/lib64' in [/usr/lib64]
    error: Bad exit status from /var/tmp/rpm-tmp.67851 (%install)
    
    
    RPM build errors:
        Bad exit status from /var/tmp/rpm-tmp.67851 (%install)
    D: May free Score board((nil))

I decided, for the moment, to silent the error by building the `RPM` with:

    $ QA_RPATHS=$[ 0x0001|0x0010 ] rpmbuild -ba -vv protobuf.spec

## Using the source RPM
The source `RPM` is very useful, since it ships all the source files, as well as the spec file, used for building a package.
The [Fedora RPM guide](http://fedoraproject.org/wiki/How_to_create_an_RPM_package) gives you more information, but you can simply run:

    $ rpm -ivh sourcepackage-name*.src.rpm

in order to install it into `~/rpmbuild`.
