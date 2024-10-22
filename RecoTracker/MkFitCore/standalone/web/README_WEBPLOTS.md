# Website Setup

Intro: Below is a short README on how to setup a CERN website to host plots produced by the benchmark and validation scripts. As this is a markdown file, it is best viewd from a web browser. However, please read the main README.md in the top-level directory prior to this README_WEBPLOTS.md! It has useful info on how to use the scripts within this directory, as well as info on how to run the benchmarking.

Apologies on the ordered list style: GitHub Flavored Markdown ignores CSS/HTML style tags, so you cannot override the list settings. Many attempts at using indents also did not work. 

## Outline

1) Setting up a website with an EOS userspace
   1) Intro
   2) Step-by-step instructions for setting up your website
   3) Setting permissions for your website
   4) Special Notes
      1) Disclaimer on email addresses
      2) Notes on ```${webdir}```
      3) Passwordless scp on LXPLUS
2) DEPRECATED: Setting up a website with an AFS userspace

## Section 1: Setting up a website with an EOS userspace
### Section 1.i: Intro

N.B. To access any of the links below, you first need to sign into CERN single login on your web browser! You will otherwise see strange errors when trying to access them.

First, ensure you have an LXPLUS account. If so, a standard ```ssh``` login into LXPLUS will take you to your /afs home directory: ```/afs/cern.ch/user/${FIRST_LETTER}/${USERNAME}```, where the first letter of your LXPLUS username is the variable ```${FIRST_LETTER}```, e.g. Kevin's LXPLUS ```${USERNAME}``` is ```kmcdermo```, so ```${FIRST_LETTER}``` is ```k```. 

Your corresponding /eos user space is: ```/eos/user/${FIRST_LETTER}/${USERNAME}```, which is allotted 1TB of storage. If you are a member of CMS, you may have additional storage on /eos at LXPLUS (through /eos/cms/store/user or /eos/cms/store/group) or the FNAL LPC. However, given the level of integration between CERNBox and the /eos/user space, it is recommended that you use the /eos/user space for storing files for the web. CERN provides now a bit more documentation on CERNBox and its connection to /eos/user space: https://cernbox-manual.web.cern.ch/cernbox-manual/en/.

In case you cannot directly access your /eos/user space, you may need to request it from CERN, following the instructions from the link above. Additional info on EOS and how to work with it can be found here: https://cern.service-now.com/service-portal/article.do?n=KB0001998. Given that EOS is still in transition, anecdotally, it is not quite as stable as AFS, and experiences some strange glitches from time to time. Always check the CERN Service Desk for incidents and planned interventions: https://cern.service-now.com/service-portal/ssb.do. In case you experience problems, open a ticket for an "Incident" through the Service Portal.

At this point, you have to determine if you are either i) looking to migrate your personal website at CERN from AFS to EOS, or ii) create a new personal website on EOS. In order to do option ii), you cannot already have a personal website with CERN. So by default, if you want to setup an website with an EOS space at CERN, and you already have an AFS website, you will need to choose option i).

### Section 1.ii: Step-by-step instructions for setting up your website 

1) Go to the instructions for setting up a website from the CERNBox documentation: https://cernbox-manual.web.cern.ch/cernbox-manual/en/web/. 
2) Follow along the steps in section 10.2: "Personal website" up until "Create personal website (via Web Services)".  
If for some reason the CERNBox documenation is down, go to this help page: https://cern.service-now.com/service-portal/article.do?n=KB0004096, and download the images on that page: eosuser-web-[1-4].png. Follow along in order of the images, as the instructions from these are equivalent to the CERNBox documentation.
3) At this point, you will need to request a website from CERN. Follow the branches i) or ii) below
   1) **Migrating your previous website from AFS to EOS**
      1) First, have a look at this link and watch the video: https://cds.cern.ch/record/2286177?ln=en, or read this document (equivalent to the video): https://indico.cern.ch/event/661564/attachments/1512296/2358778/Migrate_Website_from_AFS_to_EOS.DOCX
      2) Follow the instructions in the video, ensuring to read the text once you click on the button: "Migrate to EOS". You will have to copy your old files over to /eos/user if you want the transition to be seamless.
   2) **Brand new personal site at CERN**
       1) Continue with the CERNBox "Personal website" documenation with the section "Create personal website (via Web Services)". Or follow the instructions listed in eosuser-web-10.png from the backup help page.
4) While waiting for the request, you will need to setup directory browsing and persimissions. Some documentation on this is here: https://espace.cern.ch/webservices-help/websitemanagement/ConfiguringAFSSites/Pages/default.aspx. Please see the section below on what is recommended for restricting access to files.

N.B. Your fancy new website will have the URL: ```https://${USERNAME}.web.cern.ch/${USERNAME}/```.

### Section 1.iii: Setting permissions for your website

There are a couple options here for how to properly configure permissions for your website. At a minimum, if you just want to get your website up and running after it has been approved by CERN, login into LXPLUS and go to your website directory, i.e. ```${webdir}``` == "www" if you followed the instructions from CERNBox exactly: 

```
cd /eos/user/${FIRST_LETTER}/${USERNAME}/${webdir}
```

From there, open the file ```.htaccess``` in your favorite editor and add the following text:

```
Options +Indexes
```

However, it is recommended that your top-level directory ```${webdir}``` require at least an authenticated user sign-in to access this directory. A minimal example of what your ```.htaccess``` file needs is the text below:

```
SSLRequireSSL
AuthType shibboleth
ShibRequireSession On
ShibRequireAll On
ShibExportAssertion Off

Require valid-user
Options +Indexes
```

Upon trying to access your website now via a web browser (or another user's website with similar permissions), you will be required to sign-in via CERN's single login. If you wish to further restrict access to only members of this group, add the following line to your ```.htaccess``` file: ```Require ADFS_GROUP mic-trk-rd```. This is now setting a permission such that only members that are subscribed to our mic-trk e-group can access this directory.

**Some discussion on the ```.htaccess``` file**: If you would like to use your personal website for more than just this project, it is recommended that you create a subdirectory ```${mictrkdir}``` under ```${webdir}```. In ```${mictrkdir}```, you then can create another ```.htaccess``` file which includes the restricted access for only members of the mic-trk e-group using the line from above: ```Require ADFS_GROUP mic-trk-rd```. This line would then need to be removed in your top-level ```${webdir}/.htaccess```, in case you want others to have access to other subdirectories related to physics analysis, RECO convener duties, etc. 

### Section 1.iv: Special Notes
 
#### Section 1.iv.a: Disclaimer on email addresses
 
It is imperative that you have your primary email address associated to your CERN account (go to CERN accounts to check this) be the same email used for sending+receiving emails from the mictrk e-group. Otherwise, the line ```Require ADFS_GROUP mic-trk-rd``` will lock you out of viewing your own website on a browser! Unless you have some special CERN account, your primary email for your CERN account is a ```@cern.ch``` Outlook address. 

 
#### Section 1.iv.b: Notes on `${webdir}`
 
- If ```${webdir} != "www"```, then you will have to modify the variable ```LXPLUS_OUTDIR``` in ```web/copyAndSendToLXPLUS.sh``` to match the name for ```${webdir}```. 
- If you decided to make a subdirectory under ```${webdir}``` specifically for this project, then may wish to make the following modifications to: ```web/copyAndSendToLXPLUS.sh```
  1) Make a new variable ```LXPLUS_WEBDIR=${webdir}```, and set ```LXPLUS_OUTDIR=${mictrkdir}```.
  2) Modify the ```scp``` to be: ```scp -r ${tarball} ${LXPLUS_HOST}:${LXPLUS_WORKDIR}/${LXPLUS_WEBDIR}/${LXPLUS_OUTDIR}```
  3) Modify the ```cd``` to be: ```cd ${LXPLUS_WORKDIR}/${LXPLUS_WEBDIR}/${LXPLUS_OUTDIR}```
  4) Add this line under the untar (i.e. ```tar -zxvf```): ```cd ${LXPLUS_WORKDIR}/${LXPLUS_WEBDIR}```
 
#### Section 1.iv.c: Passwordless scp to LXPLUS
 
Make sure to read Section 10.ii.b in the main README.md on how to take advantage of passwordless scp for transferring plots to LXPLUS via ```./web/move-benchmarks.sh ${plotdir}```.

## Section 2: DEPRECATED: Setting up a website with an AFS userspace

**Special note**: This may not even be an option anymore as CERN is trying to migrate away from AFS to EOS... Therefore, instructions for this section are "as-is".

1) Request CERN website from websites.cern.ch 
   1) set website to point to AFS directory
   2) make website match username
   3) set address to ```/afs/cern.ch/user/${FIRST_LETTER}/${USERNAME}/${dir}```
   3) make sure ```${dir}``` exists!

2) While waiting for request, do the follow commands in one directory above ${dir}
   1) ```fs setacl ${dir} webserver:afs read```
   2) ```afind ${dir} -t d -e "fs setacl -dir {} -acl webserver:afs read"```
   3) ```cd ${dir}```
   4) ```touch .htaccess```
   5) open .htaccess in an editor and paste the following: ```Options +Indexes```

3) Then copy in really the very useful ```index.php``` into ```${dir}``` (optional: will simply make the top-level web GUI nice)

4) Once set up and website is live, copy plots and directories into ```${dir}```
5) ```cd ${dir}```
6) ```./makereadable.sh ${subdir}```, for every subdir. If launched from the top-level directory ```${subdir}```, it will handle the searching of subdirs.

As an aside, there are two other directories on LXPLUS every user has access to:

```
/afs/cern.ch/ubackup/${FIRST_LETTER}/${USERNAME}
```

and 

```
/afs/cern.ch/work/${FIRST_LETTER}/${USERNAME}
``` 

```ubackup``` is a backup of 24h snapshots of ```user```, while ```work``` is not backed up but users can request up to 100 GB of space. The max for ```user``` directories is 10 GB upon request.
