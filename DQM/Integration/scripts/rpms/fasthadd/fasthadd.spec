%global commit COMMITHASH

%global file_to_build fastHadd.cc
%global protobuf_message_definition ROOTFilePB.proto
%global file_for_testing test_fastHaddMerge.py
%global binary_parallel fastParallelHadd.py
%global test_driver run_fastHadd_tests.sh

%global binary_file fastHadd

Name:           fasthadd
Version:        3.0
Release:        2%{?dist}
Summary:        A program to add ProtocolBuffer-formatted ROOT files in a quick way
License:        GPLv2+
Group:          Applications/System
Source0:        %{name}-%{commit}.tar.gz
URL:            https://github.com/cms-sw/cmssw
%if 0%{?el5}
BuildRoot:      %(mktemp -ud %{_tmppath}/%{name}-%{version}-%{release}-XXXXXX)
%endif
BuildRequires:  pkgconfig
BuildRequires:  root = 5.34.18, root-tree-player = 5.34.18, root-physics = 5.34.18, root-python = 5.34.18
BuildRequires:  protobuf-devel >= 2.4.1, protobuf-compiler >= 2.4.1
Requires:       root >= 5.34.18, root-tree-player >= 5.34.18, root-physics >= 5.34.18, root-python >= 5.34.18
Requires:       protobuf >= 2.4.1

%description
A program to add ProtocolBuffer-formatted ROOT files in a quick way

%prep
%setup -q -n %{name}-%{commit}

%build
protoc -I ./ --cpp_out=./ %{protobuf_message_definition}
g++ -O2 -o %{binary_file} ROOTFilePB.pb.cc %{file_to_build} `pkg-config --libs protobuf` `root-config --cflags --libs`

#make %{?_smp_mflags}

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}%{_bindir}/
cp -p fastHadd %{buildroot}%{_bindir}/
cp -p fastParallelHadd.py %{buildroot}%{_bindir}/

%check
mkdir -p test
pushd test
cp ../%{binary_file} .
cp ../%{binary_parallel} .
cp ../%{file_for_testing} .
cp ../%{test_driver} .
export PATH=./:${PATH}
echo $PATH
. ./%{test_driver}
if [ $? -ne 0 ]; then
  exit $?
fi
popd
rm -fr test

%if 0%{?el5}
%clean
rm -rf %{buildroot}
%endif


%files
%defattr(-,root,root,-)
%doc
%{_bindir}/fastHadd
%{_bindir}/fastParallelHadd.py*

%changelog
* Tue Jul 15 2014 Dmitrijus Bugelskis <dmitrijus.bugelskis[at]cern.ch> - 3.0-2
- Fix a minor bug in merging.

* Wed Jul 09 2014 Salvatore Di Guida <salvatore.di.guida[at]cern.ch> - 3.0-1
- Use new ROOT version.
- Review not needed ROOT package dependencies (root-tree-player requires root-graf3d).
- Installation requires a version not older than the one used for building (to be investigated further).
- Better handling of files for build, install and check sections.
- Use the commit in private repository until not merged in CMSSW.

* Sat Feb 22 2014 Salvatore Di Guida <salvatore.di.guida[at]cern.ch> - 2.1-2
- Align to a fixed ROOT version.
- Ported to el6.
- Use GitHub Releases.

* Fri Nov 15 2013 Marco Rovere <marco.rovere[at]cern.ch> - 2.1-1
- Add fastParallelHadd.py to the list of files to be deployed.
- Bumped to version 2.1 and add checks while building RPMs.
