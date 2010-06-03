#ifndef Fireworks_Core_FWDialogBuilder_h
#define Fireworks_Core_FWDialogBuilder_h

#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TG3DLine.h"
#include "TGLViewer.h"
#include "TGSlider.h"
#include "TGTab.h"
#include "TGTextView.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "TGNumberEntry.h"

class FWLayoutBuilder
{
public:
   FWLayoutBuilder(TGCompositeFrame *window)
      : m_window(window),
        m_currentFrame(0),
        m_floatLeft(false),
        m_topSpacing(0),
        m_leftSpacing(0),
        m_currentHints(0)
   {
      TGVerticalFrame *mainFrame = new TGVerticalFrame(window);
      TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX|kLHintsExpandY, 
                                               0, 0, 0, 0);
      m_window->AddFrame(mainFrame, hints);
      m_framesStack.push_back(mainFrame);
      newRow();
   }
protected:
   FWLayoutBuilder &newRow()
   {
      m_currentFrame = new TGHorizontalFrame(m_framesStack.back());
      m_framesStack.back()->AddFrame(m_currentFrame, new TGLayoutHints(kLHintsExpandX));
      return *this;
   }
   
   FWLayoutBuilder &indent(int left = 2, int right = -1)
   {
      if (right < 0)
         right = left;
         
      TGVerticalFrame *parent = m_framesStack.back();
      TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX, left, right, 
                                               0, 0);
      m_currentHints = 0;
      m_framesStack.push_back(new TGVerticalFrame(parent));
      parent->AddFrame(m_framesStack.back(), hints);
      return *this;
   }
   
   /** Removes all the frames on the stack since last indent. */
   FWLayoutBuilder &unindent(void)
   {
      assert(!m_framesStack.empty());
      m_framesStack.pop_back();
      return *this;
   }

   /** Return the current layout element
     */
   TGCompositeFrame *currentFrame(void) { return m_currentFrame; }

   /** Make sure that the current layout element is going to float on the 
       left of the next one.
     */
   FWLayoutBuilder &floatLeft(size_t spacing)
   {
      m_floatLeft = true;
      m_leftSpacing = spacing;
      return *this;
   }
   
   FWLayoutBuilder &spaceDown(size_t spacing)
   {
      m_topSpacing = spacing;
      return *this;
   }
   
   /** Set whether or not the previous layout element should expand and
       in which direction.
     */
   FWLayoutBuilder &expand(bool expandX = true, bool expandY = false)
   {
      UInt_t style = 0;
      style |= expandX ? kLHintsExpandX : 0;
      style |= expandY ? kLHintsExpandY : 0;
      
      if (m_currentHints)
         m_currentHints->SetLayoutHints(style);
      return *this;
   }

protected:
   bool isFloatingLeft() { return m_floatLeft; }

   // Returns the next layout to be used.
   TGLayoutHints *nextHints()
   {
      if (m_floatLeft)
      {
         size_t left = m_leftSpacing;
         m_floatLeft = false;
         m_leftSpacing = 0;
//         if (m_currentHints)
//            m_currentHints->SetLayoutHints(kLHintsNormal);
         m_currentHints = new TGLayoutHints(kLHintsExpandX, left, 0, 
                                            m_currentHints->GetPadTop(), 0);
      }
      else
      {
         size_t top = m_topSpacing;
         m_topSpacing = 3;
         m_currentHints = new TGLayoutHints(kLHintsExpandX, 0, 0, top, 0);
      }
      return m_currentHints;
   }
   
   TGCompositeFrame *nextFrame()
   {
      if (!isFloatingLeft())
         newRow();
   
      return currentFrame();
   }
   
private:   
   TGCompositeFrame *m_window;

   std::vector<TGVerticalFrame *> m_framesStack;
   TGCompositeFrame *m_lastFrame;
   TGCompositeFrame *m_currentFrame;

   bool             m_floatLeft;
   size_t           m_topSpacing;
   size_t           m_leftSpacing;
   TGLayoutHints    *m_currentHints;
};

/** Helper class to construct dialogs in a more readable ways.

    Encapsulated TGUI layout hiccups and exposes the developer an API which
    allows to layout items in a top->bottom, right->left manner.
    
    Example:
    
      FWDialogBuilder builder(parent);
      parent.newRow(2)              // New row which has a 2 pixel padding on top.
            .addLabel("MyLabel:")    // A new label.
            .indent(20)             // Whatever follows is indented 20 pixels 
                                    // on the right.
            .addLabel("MyLabel2")   // Another label.
            .spaceDown(4)
            .addTextButton("Aligned to MyLabel2 ").floatLeft()
            .addTextButton("Same Row as previous")
            .unindent()              // back one level in the indentation.
            .addLabel("Aligned to MyLabel:")
            
    Because in ROOT layout and parenting of widgets are mixed we need to take
    responsibility for creating the widget objects (sigh!), so we have one
    "addXYZ" method per widget that can be added. If we find our way around
    this it would be better to have a generic "addWidget()" method and create
    widgets outside this class.
    
    TODO: For higher configurability we should have an 
          "addWithCallback(Callbak)"  method which can be used to specify a 
          generic widget creation action.
  */
class FWDialogBuilder : public FWLayoutBuilder
{
public:
   FWDialogBuilder(TGCompositeFrame *window, FWDialogBuilder *parent = 0)
      : FWLayoutBuilder(window),
        m_parent(parent),
        m_tabs(0)
   {}

   FWDialogBuilder &newRow()
   {
      FWLayoutBuilder::newRow();
      return *this;
   }

   FWDialogBuilder &indent(size_t indentation = 2)
   {
      FWLayoutBuilder::indent(indentation);
      return *this;
   }
   
   FWDialogBuilder &unindent(void)
   {
      FWLayoutBuilder::unindent();
      return *this;
   }
   
   FWDialogBuilder &addLabel(const char *text,
                             size_t fontSize = 12,
                             size_t weight = 0,
                             TGLabel **out = 0)
   {
      TGLabel *label = new TGLabel(nextFrame(), text);
      
      FontStruct_t defaultFontStruct = label->GetDefaultFontStruct();
      TGFontPool *pool = gClient->GetFontPool();
      TGFont* defaultFont = pool->GetFont(defaultFontStruct);
      FontAttributes_t attributes = defaultFont->GetFontAttributes();
      label->SetTextFont(pool->GetFont(attributes.fFamily, fontSize, 
                                       attributes.fWeight, attributes.fSlant));
      label->SetTextJustify(kTextLeft);
      
      currentFrame()->AddFrame(label, nextHints());
      
      return extract(label, out);
   }
   
   FWDialogBuilder &addTextView(const char *defaultText = 0,
                                TGTextView **out = 0)
   {
      TGTextView *view = new TGTextView(nextFrame(), 100, 100);
      if (defaultText)
         view->AddLine(defaultText);
      currentFrame()->AddFrame(view, nextHints());
      expand(true, true);
      return extract(view, out);
   }
   
   FWDialogBuilder &addColorPicker(const FWColorManager *manager,
                                   FWColorSelect **out = 0)
   {
      const char* graphicsLabel = " ";
      FWColorSelect *widget = new FWColorSelect(nextFrame(), graphicsLabel,
                                                0, manager, -1);
      
      currentFrame()->AddFrame(widget, nextHints());
      widget->SetEnabled(kFALSE);
      
      return extract(widget, out);
   }
   
   FWDialogBuilder &addHSlider(size_t size, TGHSlider **out = 0)
   {
      TGHSlider *slider = new TGHSlider(nextFrame(), size, kSlider1);
      currentFrame()->AddFrame(slider, nextHints());
      slider->SetRange(0, 100);
      slider->SetPosition(100);
      
      return extract(slider, out);
   }
   
   FWDialogBuilder &addTextButton(const char *text, TGTextButton **out = 0)
   {
      TGTextButton *button = new TGTextButton(nextFrame(), text);
      currentFrame()->AddFrame(button, nextHints());
      
      return extract(button, out);
   }
   
   FWDialogBuilder &addValidatingTextEntry(const char *defaultText, 
                                           FWGUIValidatingTextEntry **out)
   {
      FWGUIValidatingTextEntry *entry = new FWGUIValidatingTextEntry(nextFrame());
      currentFrame()->AddFrame(entry, nextHints());
      
      return extract(entry, out);
   }

   FWDialogBuilder &addTextEntry(const char *defaultText, 
                                 TGTextEntry **out)
   {
      TGTextEntry *entry = new TGTextEntry(nextFrame());
      currentFrame()->AddFrame(entry, nextHints());
      entry->SetEnabled(kFALSE);
      
      return extract(entry, out);
   }

   FWDialogBuilder &addNumberEntry(float defaultValue, size_t digits,
                                   TGNumberFormat::EStyle style,
                                   size_t min, size_t max,
                                   TGNumberEntry **out)
   {
      TGNumberEntry *entry = new TGNumberEntry(nextFrame(), defaultValue, 
                                               digits, -1, style,
                                               TGNumberFormat::kNEAAnyNumber,
                                               TGNumberFormat::kNELLimitMinMax,
                                               min, max);
      currentFrame()->AddFrame(entry, nextHints());
      return extract(entry, out);
   }
   
   FWDialogBuilder &addCheckbox(const char *text, TGCheckButton **out = 0)
   {
      TGCheckButton *button = new TGCheckButton(nextFrame(), text);
      button->SetState(kButtonDown, kFALSE);
      button->SetEnabled(kFALSE);
      currentFrame()->AddFrame(button, nextHints());
      
      return extract(button, out);
   }
   
   template <class T> FWDialogBuilder &extract(T *in, T **out)
   {
      if (out)
         *out = in;
      return *this;
   }
   
   FWDialogBuilder &addHSeparator(size_t horizontalPadding = 4, 
                                  size_t verticalPadding = 3)
   {
      TGLayoutHints *hints = new TGLayoutHints(kLHintsExpandX,
                                               horizontalPadding,
                                               horizontalPadding,
                                               verticalPadding,
                                               verticalPadding);

      TGHorizontal3DLine* separator = new TGHorizontal3DLine(nextFrame(), 200, 2);
      currentFrame()->AddFrame(separator, hints);
      return newRow();
   }

   /** Support for tabs.
    
       This is done by creating a new DialogBuilder and returning it for each
       of the added tabs.
       
       builder.tabs()               // Adds a TGTab widget to the current frame.
              .beginTab("Foo")      // Add a tab to the TGTab.
              .textButton("In Foo") // This is inside the tab "Foo", the layouting
                                    // is independent from previous calls
                                    // since a separate builder was returned by
                                    // 
              .endTab("Foo")        // End of the tab.
              .beginTab("Bar")
              .endTab("")
              .untabs();            // Tabs completed.
              .textButton("Main scope") // This is on the same level as the TGTab.
       
     */
   FWDialogBuilder &tabs(TGTab **out)
   {
      // Calls to tabs cannot be nested within the same builder. Multiple
      // builders are used to support nested tabs.
      assert(!m_tabs);
      m_tabs = new TGTab(nextFrame());
      currentFrame()->AddFrame(m_tabs, nextHints());
      return extract(m_tabs, out);
   }
   
   FWDialogBuilder &untabs(void)
   {
      // No untabs() without tabs().
      assert(m_tabs);
      m_tabs = 0;
      return *this;
   }
   
   /** Adds a new tab called @a label.
       A new tab gets a new builder so that tab building is completely scoped.
     */
   FWDialogBuilder &beginTab(const char *label)
   {
      TGCompositeFrame *tab = m_tabs->AddTab(label);
      FWDialogBuilder *builder = new FWDialogBuilder(tab, this);
      return builder->newRow();
   }
   
   /** When we are done with the tab, we delete ourself and return the parent.
     */
   FWDialogBuilder &endTab(void)
   {
      FWDialogBuilder *parent = m_parent;
      delete this;
      return *parent;
   }
   
   FWDialogBuilder &floatLeft(size_t spacing = 3)
   {
      FWLayoutBuilder::floatLeft(spacing);
      return *this;
   }
   
   FWDialogBuilder &spaceDown(size_t spacing = 3)
   {
      FWLayoutBuilder::spaceDown(spacing);
      return *this;
   }
   
   FWDialogBuilder &expand(size_t expandX = true, size_t expandY = false)
   {
      FWLayoutBuilder::expand(expandX, expandY);
      return *this;
   }
   
   FWDialogBuilder &vSpacer(size_t size)
   {
      FWLayoutBuilder::spaceDown(size);
      FWLayoutBuilder::newRow();
      return *this;
   }
   
private:
   FWDialogBuilder *m_parent;
   TGTab           *m_tabs;
};

#endif